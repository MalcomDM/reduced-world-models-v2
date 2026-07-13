"""Deprecated import shim for misspelled module name.

The canonical module was renamed from ``casusal_transformer`` to
``causal_transformer``.  This shim forwards imports for backward
compatibility during the transition period.

New code must import from ``rwm.models.rwm.causal_transformer`` directly.
"""

import warnings
from rwm.models.rwm.causal_transformer import CausalTransformer  # noqa: F401

warnings.warn(
    "Import 'casusal_transformer' is deprecated; use 'causal_transformer' instead.",
    DeprecationWarning,
    stacklevel=2,
)
