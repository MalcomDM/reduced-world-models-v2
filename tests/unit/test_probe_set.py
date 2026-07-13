"""Tests for deterministic probe set generation and persistence."""

import numpy as np
import pytest
from pathlib import Path
from rwm.utils.probe_set import (
    generate_probe_set,
    save_probe_set,
    load_probe_set,
    make_default_probe,
)


@pytest.mark.dataset
def test_generate_probe_set_shapes():
    obs, action = generate_probe_set(n=4, image_size=32, seed=0)
    assert obs.shape == (4, 32, 32, 3)
    assert obs.dtype == np.uint8
    assert action.shape == (4, 3)
    assert action.dtype == np.float32
    assert action[:, 0].min() >= -1.0 and action[:, 0].max() <= 1.0
    assert action[:, 1:].min() >= 0.0 and action[:, 1:].max() <= 1.0


@pytest.mark.dataset
def test_generate_probe_deterministic():
    obs1, act1 = generate_probe_set(seed=42)
    obs2, act2 = generate_probe_set(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(act1, act2)


@pytest.mark.dataset
def test_generate_probe_different_seeds_differ():
    obs1, _ = generate_probe_set(seed=0)
    obs2, _ = generate_probe_set(seed=1)
    assert not np.array_equal(obs1, obs2)


@pytest.mark.dataset
def test_save_and_load_probe(tmp_path: Path):
    obs, action = generate_probe_set(n=2, seed=0)
    path = save_probe_set(tmp_path / "probe", obs, action)
    assert path.exists()

    loaded_obs, loaded_action = load_probe_set(path)
    np.testing.assert_array_equal(obs, loaded_obs)
    np.testing.assert_array_equal(action, loaded_action)


@pytest.mark.dataset
def test_make_default_probe(tmp_path: Path):
    obs, action = make_default_probe()
    assert obs.shape == (8, 64, 64, 3)
    assert action.shape == (8, 3)

    path = tmp_path / "default_probe.npz"
    obs2, action2 = make_default_probe(path)
    assert path.exists()
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(action, action2)
