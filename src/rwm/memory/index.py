"""Stage 7.0B — Versioned Factual Pointer Index.

Provides:
- ``FactualArchive`` — ordered archive with priority computation and
  persistence
- ``EpisodeIngester`` — streaming ingestion API
- ``build_from_npz`` — deterministic builder for existing rollouts
- ``UniformSampler`` — uniform active-set control (for Stage 7.0C)
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rwm.data.split import collect_and_split
from rwm.memory.priority import (
    DEFAULT_SELECTED_H,
    DEFAULT_SELECTED_h,
    DEFAULT_CROWDING_RHO,
    DEFAULT_ETA,
    DEFAULT_LAMBDA_POS,
    DEFAULT_LAMBDA_NEG,
    DEFAULT_LAMBDA_UP,
    DEFAULT_LAMBDA_DOWN,
    DEFAULT_LAMBDA_LEGACY_DONE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_ACTIVE_SET_M,
    QUANTIZE_DECIMALS,
    EPSILON,
    quantize,
    file_hash,
    compute_factual_returns,
    compute_directional_change,
    percentile_rank,
    positive_tail_percentile,
    negative_tail_percentile,
    compute_priority_score,
    apply_equal_return_crowding,
    compute_probabilities,
    reservoir_sample,
    _stable_uniforms,
    effective_sample_size,
)
from rwm.memory.schema import (
    SCHEMA_VERSION,
    TIMING_CONTRACT_VERSION,
    make_record_id,
    FactualPointer,
    ArchiveEntry,
    PriorityConfig,
    CANONICAL_CONFIG,
)


# ---------------------------------------------------------------------------
# EpisodeIngester — streaming ingestion
# ---------------------------------------------------------------------------


class EpisodeIngester:
    """Receives transitions for one episode, computes metrics at finalization.

    Usage::

        ingester = EpisodeIngester(source_path="...", source_hash="...", ...)
        ingester.add_transition(reward=0.1, done=False)
        ingester.add_transition(reward=-0.1, done=True)
        pointers = ingester.finalize(selected_H=12, selected_h=4)
        archive.add_pointers(pointers)
    """

    def __init__(
        self,
        source_path: str,
        source_hash: str,
        data_split_seed: int,
        episode_id: str,
        source_episode_length: Optional[int] = None,
        behavior_policy: Optional[str] = None,
        dataset_manifest: str = "",
    ):
        self.source_path = source_path
        self.source_hash = source_hash
        self.data_split_seed = data_split_seed
        self.episode_id = episode_id
        self._fixed_length = source_episode_length
        self.behavior_policy = behavior_policy
        self.dataset_manifest = dataset_manifest
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._terminated: List[Optional[bool]] = []
        self._truncated: List[Optional[bool]] = []
        self._finalized = False

    def add_transition(
        self,
        reward: float,
        done: bool,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        """Append one transition without inventing unavailable termination data."""
        if self._finalized:
            raise RuntimeError("EpisodeIngester is already finalized")
        if not done and (terminated is True or truncated is True):
            raise ValueError("done=False conflicts with a true terminal flag")
        if terminated is not None and truncated is not None:
            if bool(done) != (bool(terminated) or bool(truncated)):
                raise ValueError(
                    "done must equal terminated OR truncated when both are known"
                )
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._terminated.append(
            None if terminated is None else bool(terminated)
        )
        self._truncated.append(
            None if truncated is None else bool(truncated)
        )

    def finalize(
        self,
        selected_H: int = DEFAULT_SELECTED_H,
        selected_h: int = DEFAULT_SELECTED_h,
    ) -> List[Dict[str, Any]]:
        """Compute H=12 returns and h=4 changes, return pointer dicts.

        After calling this, the ingester is frozen (no more
        ``add_transition``).
        """
        if self._finalized:
            raise RuntimeError("EpisodeIngester is already finalized")
        self._finalized = True
        T = len(self._rewards)
        if T == 0:
            return []
        rewards = np.array(self._rewards, dtype=np.float64)
        dones = np.array(self._dones, dtype=bool)
        source_ep_len = self._fixed_length if self._fixed_length is not None else T

        ret_H = compute_factual_returns(rewards, dones, selected_H)
        d_h = compute_directional_change(rewards, dones, selected_h)

        pointers = []
        for t in range(T):
            record_id = make_record_id(self.source_hash, t)
            rv = ret_H[t]
            dv = d_h[t]
            pointers.append({
                "schema_version": SCHEMA_VERSION,
                "record_id": record_id,
                "dataset_manifest": self.dataset_manifest,
                "data_split_seed": self.data_split_seed,
                "source_path": self.source_path,
                "source_hash": self.source_hash,
                "episode_id": self.episode_id,
                "timestep": t,
                "source_episode_length": source_ep_len,
                "timing_contract_version": TIMING_CONTRACT_VERSION,
                "behavior_policy": self.behavior_policy,
                "immediate_reward": float(rewards[t]),
                "legacy_done": bool(dones[t]),
                "terminated": self._terminated[t],
                "truncated": self._truncated[t],
                "factual_return_H12": (
                    None if np.isnan(rv) else round(float(rv), QUANTIZE_DECIMALS)
                ),
                "directional_change_h4": (
                    None if np.isnan(dv) else round(float(dv), QUANTIZE_DECIMALS)
                ),
            })
        return pointers


# ---------------------------------------------------------------------------
# FactualArchive
# ---------------------------------------------------------------------------


class FactualArchive:
    """Ordered archive of factual pointers with priority computation.

    Typical usage::

        archive = FactualArchive(data_split_seed=42, data_root="data/...")
        archive.add_pointers(pointer_dicts)
        archive.finalize()
        archive.save("path/to/archive.json")
        # later:
        archive2 = FactualArchive.load("path/to/archive.json")
    """

    def __init__(
        self,
        data_split_seed: int,
        data_root: str = "",
        config: Optional[PriorityConfig] = None,
    ):
        self.data_split_seed = data_split_seed
        self.data_root = data_root
        self.config = config or CANONICAL_CONFIG
        self._entries: List[Dict[str, Any]] = []
        self._finalized = False
        self._record_ids: set = set()

    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------

    @property
    def n_pointers(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return deepcopy(self._entries)

    @staticmethod
    def _pointer_payload(pointer: Dict[str, Any]) -> Dict[str, Any]:
        names = {item.name for item in fields(FactualPointer)}
        try:
            payload = {name: pointer[name] for name in names}
        except KeyError as exc:
            raise ValueError(f"Missing factual-pointer field: {exc.args[0]}") from exc
        return asdict(FactualPointer(**payload))

    def _validate_pointer(self, pointer: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._pointer_payload(pointer)
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pointer schema_version={payload['schema_version']}"
            )
        if payload["data_split_seed"] != self.data_split_seed:
            raise ValueError("Pointer data_split_seed does not match archive")
        if Path(payload["source_path"]).is_absolute():
            raise ValueError("source_path must be data-root-relative")
        if payload["timestep"] < 0:
            raise ValueError("timestep must be non-negative")
        if payload["timestep"] >= payload["source_episode_length"]:
            raise ValueError("timestep must be inside source_episode_length")
        expected_id = make_record_id(payload["source_hash"], payload["timestep"])
        if payload["record_id"] != expected_id:
            raise ValueError("record_id does not match source_hash and timestep")
        return payload

    def add_pointers(self, pointer_dicts: List[Dict[str, Any]]) -> int:
        """Insert pointer dicts idempotently. Returns number added.

        Duplicate ``record_id`` is silently skipped.  Conflicting reuse
        (same ID, different content) raises ``ValueError``.
        """
        added = 0
        for pd in pointer_dicts:
            payload = self._validate_pointer(pd)
            rid = payload["record_id"]
            if rid in self._record_ids:
                existing = next(e for e in self._entries if e["record_id"] == rid)
                if self._pointer_payload(existing) != payload:
                    raise ValueError(f"Conflicting pointer content for record_id={rid}")
                continue
            self._entries.append(payload)
            self._record_ids.add(rid)
            added += 1
        if added > 0:
            self._finalized = False
        return added

    def get_pointer(self, record_id: str) -> Optional[Dict[str, Any]]:
        for e in self._entries:
            if e["record_id"] == record_id:
                return deepcopy(e)
        return None

    def record_ids(self) -> List[str]:
        return [e["record_id"] for e in self._entries]

    # ------------------------------------------------------------------
    # Priority finalization
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Recompute all derived priority metadata from current pointers.

        After calling this, every entry contains:

        - ``qG``, ``q_pos``, ``q_neg``, ``q_up``, ``q_down``
        - ``score`` (raw priority)
        - ``crowded_score`` (after equal-return crowding)
        - ``probability`` (mixture with uniform floor)
        - ``tags`` (non-exclusive reporting list)
        - ``equal_return_group_count``
        """
        n = self.n_pointers
        if n == 0:
            self._finalized = True
            return

        cfg = self.config
        ret_arr = np.array(
            [e["factual_return_H12"] for e in self._entries], dtype=np.float64
        )
        d_arr = np.array(
            [e["directional_change_h4"] for e in self._entries], dtype=np.float64
        )
        legacy_done_arr = np.array(
            [e["legacy_done"] for e in self._entries], dtype=np.float64
        )

        qG = percentile_rank(ret_arr)
        q_pos = np.maximum(0.0, 2.0 * qG - 1.0)
        q_neg = np.maximum(0.0, 1.0 - 2.0 * qG)
        q_up = positive_tail_percentile(d_arr)
        q_down = negative_tail_percentile(d_arr)

        # Quantized return for crowding group count
        q_ret_grouped = quantize(ret_arr)
        _, inverse, counts = np.unique(
            q_ret_grouped[np.isfinite(ret_arr)],
            return_inverse=True, return_counts=True,
        )
        group_count_arr = np.full(n, 1, dtype=np.float64)
        finite_ret = np.isfinite(ret_arr)
        group_count_arr[finite_ret] = counts[inverse].astype(np.float64)

        score = compute_priority_score(
            q_pos, q_neg, q_up, q_down, legacy_done_arr,
            cfg.lambda_pos, cfg.lambda_neg, cfg.lambda_up, cfg.lambda_down,
            cfg.lambda_legacy_done, cfg.alpha, cfg.beta,
        )
        crowded = apply_equal_return_crowding(score, ret_arr, cfg.rho)
        probs = compute_probabilities(crowded, cfg.eta)

        for i, e in enumerate(self._entries):
            e["qG"] = _opt_round(qG[i])
            e["q_pos"] = _opt_round(q_pos[i])
            e["q_neg"] = _opt_round(q_neg[i])
            e["q_up"] = _opt_round(q_up[i])
            e["q_down"] = _opt_round(q_down[i])
            e["equal_return_group_count"] = int(group_count_arr[i])
            e["score"] = round(float(score[i]), 12)
            e["crowded_score"] = round(float(crowded[i]), 12)
            e["probability"] = round(float(probs[i]), 12)

            tags: List[str] = []
            if not np.isnan(q_pos[i]) and q_pos[i] > 0.5:
                tags.append("positive")
            if not np.isnan(q_neg[i]) and q_neg[i] > 0.5:
                tags.append("negative")
            if not np.isnan(q_up[i]) and q_up[i] > 0:
                tags.append("upward_surprise")
            if not np.isnan(q_down[i]) and q_down[i] > 0:
                tags.append("downward_surprise")
            if e["legacy_done"]:
                tags.append("legacy_done")
            e["tags"] = tags

        self._finalized = True

    def require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("Archive not finalized; call finalize() first")

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        self.require_finalized()
        return np.array([e["probability"] for e in self._entries], dtype=np.float64)

    def probability_summary(self) -> Dict[str, Any]:
        probs = self.probabilities()
        return {
            "n": len(probs),
            "sum": round(float(probs.sum()), 12),
            "min": round(float(probs.min()), 12),
            "max": round(float(probs.max()), 12),
            "ess": round(effective_sample_size(probs), 1),
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def uniform_sample(
        self,
        M: int,
        cycle_seed: int,
    ) -> List[str]:
        """Deterministic uniform sample without replacement.

        ``key_i = -log(U_i)`` (unit weights).  Returns sorted record_id list.
        Adding new pointers does not change existing pointers' keys.
        """
        self.require_finalized()
        n = self.n_pointers
        if M >= n:
            return self.record_ids()
        pointer_ids = tuple(self.record_ids())
        uniforms = _stable_uniforms(cycle_seed, pointer_ids)
        keys = -np.log(np.maximum(uniforms, 1e-300))
        order = np.argsort(keys)
        selected = order[:M]
        selected.sort()
        return [pointer_ids[i] for i in selected]

    def weighted_sample(
        self,
        M: int,
        cycle_seed: int,
    ) -> List[str]:
        """Deterministic weighted sample using canonical probabilities.

        This wraps ``reservoir_sample`` with the archive's probability array.
        Intended for Stage 7.1+; available here for symmetry and validation.
        """
        self.require_finalized()
        weights = self.probabilities()
        pointer_ids = tuple(self.record_ids())
        indices = reservoir_sample(weights, M, cycle_seed, pointer_ids)
        return [pointer_ids[i] for i in indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        self.require_finalized()
        return {
            "schema_version": SCHEMA_VERSION,
            "data_split_seed": self.data_split_seed,
            "data_root": self.data_root,
            "config": self.config.to_dict(),
            "n_pointers": self.n_pointers,
            "entries": deepcopy(self._entries),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Atomic JSON serialization via temp file + os.replace()."""
        self.require_finalized()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=f"archive_{self.data_split_seed}_",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=_json_fallback)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FactualArchive":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        config = PriorityConfig(**data.get("config", {}))
        archive = cls(
            data_split_seed=data["data_split_seed"],
            data_root=data.get("data_root", ""),
            config=config,
        )
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported archive schema_version={data.get('schema_version')}"
            )
        archive.add_pointers(data["entries"])
        # Restore only recomputable priority metadata after pointer validation.
        for target, source in zip(archive._entries, data["entries"]):
            target.update({
                key: deepcopy(value)
                for key, value in source.items()
                if key not in archive._pointer_payload(source)
            })
        archive._finalized = True
        return archive

    def validate_sources(
        self,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Reject missing or content-mismatched factual source files."""
        root = Path(self.data_root if data_root is None else data_root)
        checked: Dict[str, str] = {}
        for entry in self._entries:
            rel_path = entry["source_path"]
            expected_hash = entry["source_hash"]
            if rel_path in checked:
                if checked[rel_path] != expected_hash:
                    raise ValueError(f"Conflicting source hashes for {rel_path}")
                continue
            source = root / rel_path
            if not source.is_file():
                raise ValueError(f"Missing factual source: {source}")
            actual_hash = file_hash(source)
            if actual_hash != expected_hash:
                raise ValueError(f"Source hash mismatch: {source}")
            checked[rel_path] = expected_hash

    # ------------------------------------------------------------------
    # Digest
    # ------------------------------------------------------------------

    def digest(self) -> str:
        """SHA-256 of canonical JSON output — stable across identical archives."""
        import hashlib
        as_json = json.dumps(
            self.to_dict(), sort_keys=True, default=_json_fallback,
        )
        return hashlib.sha256(as_json.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def train_val_disjointness(self, val_ids: set) -> bool:
        self.require_finalized()
        record_sources = {e["source_hash"] for e in self._entries}
        return record_sources.isdisjoint(val_ids)


def _opt_round(x: float) -> Optional[float]:
    if np.isnan(x):
        return None
    return round(float(x), 8)


# ---------------------------------------------------------------------------
# NPZ builder
# ---------------------------------------------------------------------------


def build_from_npz(
    data_root: Union[str, Path],
    data_split_seed: int,
    selected_H: int = DEFAULT_SELECTED_H,
    selected_h: int = DEFAULT_SELECTED_h,
    config: Optional[PriorityConfig] = None,
) -> FactualArchive:
    """Deterministic archive builder from existing NPZ rollouts.

    Loads only ``reward`` and ``done`` arrays from each training file.
    Never decodes or copies observation arrays.  Validation files are
    rejected by ``collect_and_split``.
    """
    data_root = Path(data_root)
    config = config or CANONICAL_CONFIG
    if selected_H != config.selected_H or selected_h != config.selected_h:
        raise ValueError(
            "selected_H/selected_h must match the persisted PriorityConfig"
        )
    archive = FactualArchive(
        data_split_seed=data_split_seed,
        data_root=str(data_root),
        config=config,
    )
    train_files, val_files = collect_and_split(data_root, data_split_seed)

    # Verify disjointness
    train_set = set(train_files)
    val_set = set(val_files)
    if not train_set.isdisjoint(val_set):
        raise ValueError("Train/val file sets overlap")

    for fpath in sorted(train_files):
        fhash = file_hash(fpath)
        with np.load(fpath) as data:
            rewards = np.asarray(data["reward"])
            dones = np.asarray(data["done"])
            T = len(rewards)

            ingester = EpisodeIngester(
                source_path=str(fpath.relative_to(data_root)),
                source_hash=fhash,
                data_split_seed=data_split_seed,
                episode_id=fpath.stem,
                source_episode_length=T,
                behavior_policy="random",
                dataset_manifest=f"collect_and_split(seed={data_split_seed})",
            )
            for t in range(T):
                ingester.add_transition(
                    reward=float(rewards[t]),
                    done=bool(dones[t]),
                )
        pointers = ingester.finalize(
            selected_H=selected_H, selected_h=selected_h,
        )
        archive.add_pointers(pointers)

    archive.finalize()
    return archive


# ---------------------------------------------------------------------------
# UniformSampler — 7.0C control
# ---------------------------------------------------------------------------


class UniformSampler:
    """Deterministic uniform active-set sampler for Stage 7.0C.

    Wraps ``FactualArchive.uniform_sample`` with configurable M and
    stable metadata provenance.  Does NOT use priority weights.
    """

    def __init__(self, archive: FactualArchive, M: int = DEFAULT_ACTIVE_SET_M):
        self.archive = archive
        self.M = M
        self._last_sample: Optional[Dict[str, Any]] = None

    def sample(self, cycle_seed: int) -> List[str]:
        ids = self.archive.uniform_sample(self.M, cycle_seed)
        self._last_sample = {
            "M": self.M,
            "cycle_seed": cycle_seed,
            "n_archive": self.archive.n_pointers,
            "record_ids": ids,
            "sampler": "uniform_gumbel",
        }
        return ids

    @property
    def last_sample(self) -> Optional[Dict[str, Any]]:
        return self._last_sample


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_fallback(obj: Any) -> str:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)
