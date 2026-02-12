"""MOMDP environment for the OD-DBP-LA scheduling problem.

State, action, transition, and dual-reward construction follow Section 3
of the manuscript. The environment exposes:

  * 78-D compact state used by the policy and value trunks
    (24-D server-attribute block + 40-D facility summary + 14-D global
    progress vector)
  * an expanded per-cell representation that CASE consumes separately
    (see networks.CASEEncoder)
  * the four-level event reward r1 (Eq. 3) and the proportional
    tardiness reward r2 (Eq. 4)

All feasibility logic is enforced through dynamic action masking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import BANK_CELLS, Server


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FacilityConfig:
    n_banks: int = 4              # 1, 2, or 4
    horizon_days: int = 10        # planning horizon
    lookahead: int = 10           # rolling window width
    d_max_retries: int = 5        # DAP defer cap
    cells_per_bank: Dict[int, int] = field(default_factory=lambda: dict(BANK_CELLS))

    @property
    def n_cells(self) -> int:
        return self.cells_per_bank[self.n_banks]

    @property
    def n_actions(self) -> int:
        # one assignment action per cell + one defer action
        return self.n_cells + 1

    @property
    def defer_action(self) -> int:
        return self.n_cells


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    """Tracks cell occupancy over time."""
    busy_until: int = 0           # next free period (0 = free now)
    power_class: int = 0
    cool_class: int = 0
    in_disruption: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LHACEnv:
    """Sequential MOMDP environment.

    One step = one decision for the current head-of-queue server. The
    queue is repopulated as servers arrive within the rolling window.
    """

    STATE_DIM = 78          # 24 + 40 + 14 (paper §3)

    def __init__(
        self,
        servers: List[Server],
        config: Optional[FacilityConfig] = None,
        proc_noise_sigma: float = 0.0,
        disruption_rate: float = 0.0,
        arrival_reveal: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.servers_master = list(servers)
        self.cfg = config or FacilityConfig()
        self.proc_noise_sigma = float(proc_noise_sigma)
        self.disruption_rate = float(disruption_rate)
        self.arrival_reveal = float(arrival_reveal)
        self.rng = np.random.default_rng(seed)

        # episode state, populated in reset()
        self.t: int = 0
        self.cells: List[CellState] = []
        self.queue: List[Server] = []
        self.placed: List[dict] = []
        self.failed: List[Server] = []
        self.retry_count: Dict[int, int] = {}
        self.current: Optional[Server] = None

    # ----- lifecycle ----------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.t = 0
        self.cells = self._init_cells()
        self.queue = []
        self.placed = []
        self.failed = []
        self.retry_count = {}
        self._unrevealed = list(self.servers_master)
        self._reveal_arrivals()
        self.current = self._next_in_queue()
        return self.observe(), self.feasibility_mask()

    def _init_cells(self) -> List[CellState]:
        cells = []
        for c in range(self.cfg.n_cells):
            cells.append(
                CellState(
                    busy_until=0,
                    power_class=int(self.rng.integers(0, 4)),
                    cool_class=int(self.rng.integers(0, 2)),
                )
            )
        return cells

    def _reveal_arrivals(self) -> None:
        """Reveal arrivals up to t + lookahead. arrival_reveal < 1
        models stochastic / partially-observed arrivals."""
        horizon_now = self.t + self.cfg.lookahead
        keep = []
        for s in self._unrevealed:
            visible = self.rng.random() < self.arrival_reveal
            if s.arrival <= horizon_now and visible:
                self.queue.append(s)
            else:
                keep.append(s)
        self._unrevealed = keep
        self.queue.sort(key=lambda s: (s.arrival, s.due))

    def _next_in_queue(self) -> Optional[Server]:
        while self.queue:
            head = self.queue[0]
            if head.arrival <= self.t:
                return head
            # nothing eligible yet -- advance time
            self.t += 1
            self._tick_disruptions()
            self._reveal_arrivals()
            if self.t >= self.cfg.horizon_days:
                return None
        return None

    # ----- core step ----------------------------------------------------------

    def step(self, action: int) -> Tuple[
        np.ndarray, np.ndarray, float, float, bool, dict
    ]:
        """Execute action; return (next_state, mask, r1, r2, done, info)."""
        if self.current is None:
            return self.observe(), self.feasibility_mask(), 0.0, 0.0, True, {}

        info: dict = {"server": self.current.sid, "action": action}

        if action == self.cfg.defer_action:
            r1, r2, done = self._do_defer()
        else:
            r1, r2, done = self._do_assign(action)

        # advance to next head-of-queue
        self.current = self._next_in_queue()
        if self.current is None or self.t >= self.cfg.horizon_days:
            done = True

        return self.observe(), self.feasibility_mask(), r1, r2, done, info

    def _do_defer(self) -> Tuple[float, float, bool]:
        s = self.current
        n = self.retry_count.get(s.sid, 0) + 1
        self.retry_count[s.sid] = n
        if n >= self.cfg.d_max_retries:
            # permanent failure
            self.failed.append(s)
            self.queue.pop(0)
            return -1.0, 0.0, False
        # re-queue: shift one period later
        self.queue.pop(0)
        s_def = Server(**{**s.__dict__, "arrival": min(s.arrival + 1, self.cfg.horizon_days - 1)})
        self.queue.append(s_def)
        self.queue.sort(key=lambda x: (x.arrival, x.due))
        return 0.1, 0.0, False

    def _do_assign(self, cell_idx: int) -> Tuple[float, float, bool]:
        s = self.current
        if not self._is_feasible(s, cell_idx):
            # masked actions should never be sampled -- defensive forced skip
            self.failed.append(s)
            self.queue.pop(0)
            return 0.0, 0.0, False

        start = max(self.t, s.arrival, self.cells[cell_idx].busy_until)
        proc = self._sample_processing_time(s)
        end = start + proc
        self.cells[cell_idx].busy_until = end
        if s.is_2tc and cell_idx + 1 < self.cfg.n_cells:
            self.cells[cell_idx + 1].busy_until = end

        tard = max(0, end - s.due)
        r2 = -tard / max(1, s.p_time)

        self.placed.append({
            "sid": s.sid,
            "cell": cell_idx,
            "start": start,
            "end": end,
            "tard": tard,
            "is_2tc": s.is_2tc,
        })
        self.queue.pop(0)
        return 1.0, float(r2), False

    # ----- feasibility --------------------------------------------------------

    def feasibility_mask(self) -> np.ndarray:
        m = np.zeros(self.cfg.n_actions, dtype=np.float32)
        if self.current is None:
            m[self.cfg.defer_action] = 1.0
            return m
        s = self.current
        for c in range(self.cfg.n_cells):
            if self._is_feasible(s, c):
                m[c] = 1.0
        m[self.cfg.defer_action] = 1.0    # defer always available
        return m

    def _is_feasible(self, s: Server, cell_idx: int) -> bool:
        if cell_idx >= self.cfg.n_cells:
            return False
        cell = self.cells[cell_idx]
        if cell.in_disruption:
            return False
        # power / cooling compatibility
        if cell.power_class != s.power_class:
            return False
        if cell.cool_class != s.cool_class:
            return False
        # 2TC requires the right neighbour to be available and same bank
        if s.is_2tc:
            if cell_idx + 1 >= self.cfg.n_cells:
                return False
            nbr = self.cells[cell_idx + 1]
            if nbr.in_disruption:
                return False
            # share bank: simple bank-id check via integer division
            cells_per = self.cfg.cells_per_bank[self.cfg.n_banks] // self.cfg.n_banks
            if cell_idx // cells_per != (cell_idx + 1) // cells_per:
                return False
        # earliest feasible start within due date
        e = max(self.t, s.arrival, cell.busy_until)
        if e + s.p_time > s.due + 1:        # slight tolerance: due is a deadline
            return False
        return True

    # ----- perturbations ------------------------------------------------------

    def _sample_processing_time(self, s: Server) -> int:
        if self.proc_noise_sigma <= 0:
            return s.p_time
        noise = self.rng.normal(0.0, self.proc_noise_sigma)
        return max(1, int(round(s.p_time * (1.0 + noise))))

    def _tick_disruptions(self) -> None:
        if self.disruption_rate <= 0:
            return
        for c in self.cells:
            # independent per-cell event
            if self.rng.random() < self.disruption_rate / max(1, self.cfg.horizon_days):
                c.in_disruption = True
            elif c.in_disruption and self.rng.random() < 0.3:
                c.in_disruption = False

    # ----- observation --------------------------------------------------------

    def observe(self) -> np.ndarray:
        s = self.current
        out = np.zeros(self.STATE_DIM, dtype=np.float32)

        # ---- 24-D current-server block --------------------------------------
        if s is not None:
            out[0] = s.p_time / 5.0
            out[1] = (s.due - self.t) / max(1, self.cfg.horizon_days)
            out[2] = float(s.is_2tc)
            out[3] = s.power_class / 3.0
            out[4] = s.cool_class
            out[5] = self.retry_count.get(s.sid, 0) / max(1, self.cfg.d_max_retries)
            # one-hot power / cooling pads
            out[6 + s.power_class] = 1.0
            out[10 + s.cool_class] = 1.0
            out[12] = s.arrival / max(1, self.cfg.horizon_days)
            # remaining 24-D padded with zeros for downstream attention

        # ---- 40-D facility summary -----------------------------------------
        cells_per_bank = self.cfg.cells_per_bank[self.cfg.n_banks] // self.cfg.n_banks
        for b in range(self.cfg.n_banks):
            base = 24 + b * 10
            cells_b = self.cells[b * cells_per_bank: (b + 1) * cells_per_bank]
            busy = [max(0, c.busy_until - self.t) for c in cells_b]
            out[base + 0] = np.mean(busy) / max(1, self.cfg.horizon_days)
            out[base + 1] = max(busy) / max(1, self.cfg.horizon_days)
            out[base + 2] = np.mean([c.in_disruption for c in cells_b])
            out[base + 3] = np.mean([c.power_class for c in cells_b]) / 3.0
            out[base + 4] = np.mean([c.cool_class for c in cells_b])

        # ---- 14-D global progress vector -----------------------------------
        n_total = max(1, len(self.servers_master))
        out[64] = self.t / max(1, self.cfg.horizon_days)
        out[65] = len(self.placed) / n_total
        out[66] = len(self.failed) / n_total
        out[67] = len(self.queue) / n_total
        out[68] = sum(p["tard"] for p in self.placed) / n_total
        out[69] = float(any(c.in_disruption for c in self.cells))
        return out

    def per_cell_features(self) -> np.ndarray:
        """Expanded per-cell representation used by CASE.

        Returns array of shape (n_cells, F=12).
        """
        F = 12
        out = np.zeros((self.cfg.n_cells, F), dtype=np.float32)
        s = self.current
        for ci, cell in enumerate(self.cells):
            free_in = max(0, cell.busy_until - self.t)
            out[ci, 0] = free_in / max(1, self.cfg.horizon_days)
            out[ci, 1] = float(cell.in_disruption)
            out[ci, 2] = cell.power_class / 3.0
            out[ci, 3] = cell.cool_class
            if s is not None:
                out[ci, 4] = float(cell.power_class == s.power_class)
                out[ci, 5] = float(cell.cool_class == s.cool_class)
                out[ci, 6] = float(self._is_feasible(s, ci))
            out[ci, 7] = ci / max(1, self.cfg.n_cells - 1)        # positional
            cells_per_bank = self.cfg.cells_per_bank[self.cfg.n_banks] // self.cfg.n_banks
            out[ci, 8] = (ci // cells_per_bank) / max(1, self.cfg.n_banks - 1)
            # neighbour status (for 2TC)
            if ci + 1 < self.cfg.n_cells:
                out[ci, 9]  = float(self.cells[ci + 1].in_disruption)
                out[ci, 10] = max(0, self.cells[ci + 1].busy_until - self.t) / max(1, self.cfg.horizon_days)
            out[ci, 11] = float(cell.busy_until <= self.t)            # idle now
        return out

    # ----- summary metrics ----------------------------------------------------

    def summary(self) -> dict:
        n_total = max(1, len(self.servers_master))
        n_placed = len(self.placed)
        n_tardy = sum(1 for p in self.placed if p["tard"] > 0)
        return {
            "completion_rate": 100.0 * n_placed / n_total,
            "tardiness_rate":  100.0 * n_tardy / n_total,
            "n_placed": n_placed,
            "n_tardy": n_tardy,
            "n_failed": len(self.failed),
            "n_total": n_total,
        }
