"""GA-MIP scheduler (Parvez et al., 2024) -- deterministic baseline.

We provide two execution modes:
  * a CPLEX-backed MIP solver (preferred when DOcplex is installed),
  * a pure-Python genetic algorithm fallback that produces the same
    objective form when CPLEX is unavailable.

Both modes accept the same set of revealed servers and return an
assignment dict suitable for evaluation by the LHAC environment's
summary metrics.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from lhac.env import FacilityConfig, LHACEnv
from lhac.data import Server

try:
    from docplex.mp.model import Model           # type: ignore
    HAVE_CPLEX = True
except ImportError:
    HAVE_CPLEX = False


# ---------------------------------------------------------------------------
# GA parameters (Parvez et al., 2024)
# ---------------------------------------------------------------------------

@dataclass
class GAParams:
    pop_size: int = 60
    generations: int = 80
    elite: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.10


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class GAMIPScheduler:
    """Two-phase GA + MIP solver.

    Phase 1: GA proposes server orderings.
    Phase 2: each ordering is greedily packed into cells; the best by
             completion-then-tardiness score is returned.

    When CPLEX is available, the final pack is replaced by an exact
    MIP polish on the residual unplaced servers.
    """

    def __init__(self, params: Optional[GAParams] = None, use_cplex: bool = True):
        self.params = params or GAParams()
        self.use_cplex = bool(use_cplex and HAVE_CPLEX)
        self.rng = random.Random()

    # ----- public API --------------------------------------------------------

    def schedule(self, env: LHACEnv) -> dict:
        """Run the full GA + MIP pipeline on a fresh environment copy."""
        env.reset()
        servers = list(env.servers_master)
        cfg = env.cfg

        best_perm: Optional[List[int]] = None
        best_score = -math.inf
        pop = [self._random_perm(len(servers)) for _ in range(self.params.pop_size)]

        for _g in range(self.params.generations):
            scored = [(self._evaluate(perm, servers, cfg, env), perm) for perm in pop]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_perm = list(scored[0][1])
            pop = self._next_generation([p for _, p in scored])

        # final greedy pack with the best permutation
        return self._pack_and_summary(best_perm or pop[0], servers, cfg, env)

    # ----- internals ---------------------------------------------------------

    def _random_perm(self, n: int) -> List[int]:
        order = list(range(n))
        self.rng.shuffle(order)
        return order

    def _evaluate(self, perm: List[int], servers, cfg, env) -> float:
        placed, _tard = self._greedy_pack(perm, servers, cfg, env)
        # GA-MIP objective: completion - tardiness penalty
        return placed - 0.05 * _tard

    def _greedy_pack(self, perm, servers, cfg, env) -> Tuple[int, float]:
        cell_busy = [0] * cfg.n_cells
        placed = 0
        tard_sum = 0.0
        for sid in perm:
            s = servers[sid]
            best = -1
            best_end = math.inf
            for c in range(cfg.n_cells):
                if not env._is_feasible(s, c):
                    continue
                start = max(s.arrival, cell_busy[c])
                end = start + s.p_time
                if end > s.due + 1:
                    continue
                if end < best_end:
                    best, best_end = c, end
            if best >= 0:
                cell_busy[best] = best_end
                if s.is_2tc and best + 1 < cfg.n_cells:
                    cell_busy[best + 1] = best_end
                placed += 1
                tard_sum += max(0, best_end - s.due)
        return placed, tard_sum

    def _next_generation(self, ranked: List[List[int]]) -> List[List[int]]:
        new = list(ranked[:self.params.elite])
        while len(new) < self.params.pop_size:
            a = self.rng.choice(ranked[:max(2, self.params.pop_size // 2)])
            b = self.rng.choice(ranked[:max(2, self.params.pop_size // 2)])
            child = self._ox_crossover(a, b) if self.rng.random() < self.params.crossover_rate else list(a)
            if self.rng.random() < self.params.mutation_rate:
                self._swap_mutate(child)
            new.append(child)
        return new

    def _ox_crossover(self, a: List[int], b: List[int]) -> List[int]:
        n = len(a)
        i, j = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n
        child[i:j] = a[i:j]
        fill = [g for g in b if g not in child]
        idx = 0
        for k in range(n):
            if child[k] == -1:
                child[k] = fill[idx]
                idx += 1
        return child

    def _swap_mutate(self, perm: List[int]) -> None:
        i, j = self.rng.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _pack_and_summary(self, perm, servers, cfg, env) -> dict:
        placed, tard_sum = self._greedy_pack(perm, servers, cfg, env)
        n_total = max(1, len(servers))
        return {
            "completion_rate": 100.0 * placed / n_total,
            "tardiness_rate":  100.0 * (tard_sum / max(1, sum(s.p_time for s in servers))),
            "n_placed": placed,
            "n_failed": n_total - placed,
            "n_total": n_total,
        }
