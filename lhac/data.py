"""Synthetic and benchmark dataset generation for the server-to-cell
scheduling problem (OD-DBP-LA, Parvez et al. 2024).

The deterministic instance follows three arrival patterns (Uniform,
Right-skewed, Left-skewed), three facility sizes (200, 300, 400 servers),
and three two-test-cell ratios (10%, 20%, 30%).
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-server record
# ---------------------------------------------------------------------------

@dataclass
class Server:
    sid: int                # server identifier
    arrival: int            # arrival period (days from t=0)
    p_time: int             # processing time (periods)
    due: int                # due date (period)
    is_2tc: int             # 1 if requires two adjacent cells, else 0
    power_class: int        # power compatibility tag (0..3)
    cool_class: int         # cooling compatibility tag (0..1)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DataGenerator:
    """Generate synthetic OD-DBP-LA instances calibrated against the
    industry-partner distributions used in the paper.

    Parameters
    ----------
    pattern : {"uniform", "right_skewed", "left_skewed"}
    n_servers : int
    twotc_ratio : float in [0, 1]
    horizon_days : int
    seed : int
    """

    PATTERNS = ("uniform", "right_skewed", "left_skewed")

    def __init__(
        self,
        pattern: str = "uniform",
        n_servers: int = 300,
        twotc_ratio: float = 0.20,
        horizon_days: int = 10,
        seed: Optional[int] = None,
    ):
        if pattern not in self.PATTERNS:
            raise ValueError(f"pattern must be one of {self.PATTERNS}")
        self.pattern = pattern
        self.n_servers = int(n_servers)
        self.twotc_ratio = float(twotc_ratio)
        self.horizon = int(horizon_days)
        self.rng = np.random.default_rng(seed)

    # --- arrival distribution -------------------------------------------------

    def _arrival_pdf(self) -> np.ndarray:
        """Discrete probability mass over days [0..horizon-1]."""
        x = np.arange(self.horizon)
        if self.pattern == "uniform":
            w = np.ones(self.horizon)
        elif self.pattern == "right_skewed":
            # heavier mass early in horizon (more arrivals up front)
            w = np.exp(-x / max(2.0, self.horizon / 4.0))
        else:  # left_skewed
            w = np.exp(-(self.horizon - 1 - x) / max(2.0, self.horizon / 4.0))
        return w / w.sum()

    # --- main entry -----------------------------------------------------------

    def build(self) -> List[Server]:
        pdf = self._arrival_pdf()
        arrivals = self.rng.choice(self.horizon, size=self.n_servers, p=pdf)

        # processing time: 1..3 days, mode at 2
        p_times = self.rng.choice([1, 2, 3], size=self.n_servers, p=[0.25, 0.55, 0.20])

        # due dates: arrival + processing + slack (slack ~ U[0..3])
        slack = self.rng.integers(0, 4, size=self.n_servers)
        dues = np.minimum(arrivals + p_times + slack, self.horizon - 1)

        # 2TC flag
        n_2tc = int(round(self.n_servers * self.twotc_ratio))
        is_2tc = np.zeros(self.n_servers, dtype=int)
        is_2tc[:n_2tc] = 1
        self.rng.shuffle(is_2tc)

        # power / cooling compatibility tags
        power = self.rng.integers(0, 4, size=self.n_servers)
        cool = self.rng.integers(0, 2, size=self.n_servers)

        servers = []
        for i in range(self.n_servers):
            servers.append(
                Server(
                    sid=i,
                    arrival=int(arrivals[i]),
                    p_time=int(p_times[i]),
                    due=int(dues[i]),
                    is_2tc=int(is_2tc[i]),
                    power_class=int(power[i]),
                    cool_class=int(cool[i]),
                )
            )
        # sort by arrival, then by due date
        servers.sort(key=lambda s: (s.arrival, s.due))
        for new_sid, s in enumerate(servers):
            s.sid = new_sid
        return servers

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.as_dict() for s in self.build()])


# ---------------------------------------------------------------------------
# Facility configuration helpers
# ---------------------------------------------------------------------------

BANK_CELLS = {1: 14, 2: 28, 4: 54}    # cells per bank-config used in paper

def cells_for_banks(n_banks: int) -> int:
    if n_banks not in BANK_CELLS:
        raise ValueError(f"n_banks must be one of {list(BANK_CELLS.keys())}")
    return BANK_CELLS[n_banks]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_dataset(
    pattern: str = "uniform",
    n_servers: int = 300,
    twotc_ratio: float = 0.20,
    horizon_days: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """One-call dataset generation that returns a pandas DataFrame."""
    return DataGenerator(
        pattern=pattern,
        n_servers=n_servers,
        twotc_ratio=twotc_ratio,
        horizon_days=horizon_days,
        seed=seed,
    ).as_dataframe()


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV instance previously written by generate_dataset()."""
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Pattern grid utility (used by reproduce/ scripts)
# ---------------------------------------------------------------------------

PATTERNS_GRID = ("uniform", "right_skewed", "left_skewed")
SIZES_GRID = (200, 300, 400)
TWOTC_GRID = (0.10, 0.20, 0.30)

def benchmark_grid() -> Iterable[Tuple[str, int, float]]:
    for p in PATTERNS_GRID:
        for n in SIZES_GRID:
            for r in TWOTC_GRID:
                yield p, n, r
