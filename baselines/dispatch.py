"""Classical dispatching heuristics: EDD and Slack-based.

Both follow the same single-pass loop: at every decision step, place the
head-of-queue server in the first feasible cell ranked by the chosen
priority rule. No learning, no lookahead optimization.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from lhac.env import FacilityConfig, LHACEnv


def _run_dispatch(env: LHACEnv, score: Callable[[LHACEnv, int], float]) -> dict:
    state, mask = env.reset()
    done = False
    while not done:
        feas = [i for i, v in enumerate(mask) if v > 0 and i != env.cfg.defer_action]
        if not feas:
            a = env.cfg.defer_action
        else:
            a = min(feas, key=lambda i: score(env, i))
        state, mask, _r1, _r2, done, _info = env.step(a)
    return env.summary()


class EDDScheduler:
    """Earliest-Due-Date dispatching: prefer cells that minimise late finish."""

    def schedule(self, env: LHACEnv) -> dict:
        return _run_dispatch(env, lambda e, c: e.current.due if e.current else 0)


class SlackScheduler:
    """Minimum-slack dispatching: slack = due - (now + processing time)."""

    def schedule(self, env: LHACEnv) -> dict:
        def slack(e: LHACEnv, c: int) -> float:
            s = e.current
            if s is None:
                return 0.0
            return float(s.due - (e.t + s.p_time))
        return _run_dispatch(env, slack)
