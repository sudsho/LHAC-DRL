"""Feasibility-Preserving Reward shaping (FPR).

Given Agent 1's base completion reward r1 (Eq. 3) and the feasibility
potential phi(s) = |{k in U(s) : exists feasible cell}| / |U(s)|
(Eq. 8), the shaped reward consumed by Agent 1 is

    r_tilde_1 = r1 + gamma * phi(s') - phi(s)

(Eq. 9). The shaping term is potential-based and therefore preserves
the primary-objective preference structure under deterministic
transitions (Ng et al., 1999). Agent 2 never observes this term.
"""
from __future__ import annotations

from typing import List

from .env import LHACEnv


def feasibility_potential(env: LHACEnv) -> float:
    """phi(s) = fraction of currently unassigned servers that retain at
    least one feasible cell at state s."""
    n_unassigned = len(env.queue) + (1 if env.current is not None else 0)
    if n_unassigned == 0:
        return 0.0
    survivors = 0
    candidates = list(env.queue)
    if env.current is not None:
        candidates = [env.current] + candidates
    for s in candidates:
        for c in range(env.cfg.n_cells):
            if env._is_feasible(s, c):
                survivors += 1
                break
    return survivors / n_unassigned


def shaped_reward(r1: float, phi_s: float, phi_s_next: float, gamma: float) -> float:
    """Eq. (9): r_tilde_1 = r1 + gamma * phi(s') - phi(s)."""
    return float(r1) + gamma * float(phi_s_next) - float(phi_s)
