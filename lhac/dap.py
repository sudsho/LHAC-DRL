"""Deferred Action Placement (DAP).

DAP augments the action space with a defer action so the scheduler can
postpone the head-of-queue server when revealed capacity is tight but
future arrivals within the lookahead may create feasible openings.

The defer logic is enforced inside `LHACEnv` (see env.py); this module
provides a small helper to construct the augmented mask and to query
the per-server retry budget defined by D_max in Section 4.2.1.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .env import LHACEnv


def augmented_mask(env: LHACEnv) -> np.ndarray:
    """Return the feasibility mask with the defer slot always enabled
    (defer is the n_cells-th index)."""
    return env.feasibility_mask()


def can_defer(env: LHACEnv) -> bool:
    """True iff the current server has not yet exhausted its retry
    budget D_max."""
    if env.current is None:
        return False
    n = env.retry_count.get(env.current.sid, 0)
    return n < env.cfg.d_max_retries


def retry_budget(env: LHACEnv) -> Dict[int, int]:
    """Remaining retries per server still in flight."""
    out = {}
    if env.current is not None:
        out[env.current.sid] = env.cfg.d_max_retries - env.retry_count.get(env.current.sid, 0)
    for s in env.queue:
        out[s.sid] = env.cfg.d_max_retries - env.retry_count.get(s.sid, 0)
    return out
