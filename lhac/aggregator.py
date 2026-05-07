"""Loader for the per-seed experiment logs under `results/`.

Each completed run of the LHAC training and evaluation pipelines
appends one row per (method, instance, seed) tuple to a flat CSV log
file. The classes in this module read those logs and expose a small
lookup API that the reproduction scripts use to aggregate outcomes
across seeds without re-running the underlying experiments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    """A single seed's outcome on one (method, instance) configuration."""
    completion: float
    tardiness: float
    runtime_sec: float = 0.0


# ---------------------------------------------------------------------------
# Pattern normaliser
# ---------------------------------------------------------------------------

_PATTERN_NORM = {
    'uniform': 'Uniform', 'right_skewed': 'Right-skewed',
    'left_skewed': 'Left-skewed', 'Uniform': 'Uniform',
    'Right-skewed': 'Right-skewed', 'Left-skewed': 'Left-skewed',
}

def _pat(p): return _PATTERN_NORM.get(p, p)
def _twotc(t): return f'{int(round(float(t) * 100))}%'


# ---------------------------------------------------------------------------
# Per-seed log loader
# ---------------------------------------------------------------------------

class ExperimentLog:
    """Reads the per-seed experiment logs written during the training
    and evaluation sweeps. Provides an addressable lookup keyed by
    (method, instance, seed) so that downstream analysis can pull
    individual outcomes without rerunning the underlying solver."""

    def __init__(self, results_dir: Optional[str] = None):
        rd = results_dir or RESULTS
        self._main = pd.read_csv(os.path.join(rd, 'main_per_seed.csv'))
        self._rob  = pd.read_csv(os.path.join(rd, 'robustness_per_seed.csv'))
        self._abl  = pd.read_csv(os.path.join(rd, 'ablation_components_per_seed.csv'))
        self._tlo  = pd.read_csv(os.path.join(rd, 'ablation_tlo_per_seed.csv'))
        self._rs   = pd.read_csv(os.path.join(rd, 'ablation_reward_scale_per_seed.csv'))
        self._morl = pd.read_csv(os.path.join(rd, 'morl_per_seed.csv'))
        self._arch = pd.read_csv(os.path.join(rd, 'architecture_per_seed.csv'))

    # ----- main-comparison lookup -----
    def main_comparison(self, method: str, pattern: str, size: int,
                        twotc: float, seed: int) -> Metric:
        df = self._main
        row = df[(df['pattern'] == _pat(pattern)) &
                 (df['size'] == size) &
                 (df['twotc'] == _twotc(twotc)) &
                 (df['method'] == method.lower()) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for ({method}, {pattern}, {size}, {twotc}, seed={seed})')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']),
                      runtime_sec=float(r.get('runtime_sec', 0.0)))

    # ----- robustness lookup -----
    def robustness(self, method: str, perturb_type: str, level: float,
                   seed: int, pattern: str = 'Uniform',
                   size: int = 200, twotc: str = '10%') -> Metric:
        df = self._rob
        row = df[(df['perturb_type'] == perturb_type) &
                 (df['level'].astype(float).round(6) == round(float(level), 6)) &
                 (df['method'] == method.lower()) &
                 (df['pattern'] == pattern) &
                 (df['size'] == size) &
                 (df['twotc'] == twotc) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for {(perturb_type, level, method, seed)}')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']))

    def robustness_aggregated(self, perturb_type: str, level: float,
                              method: str = 'lhac') -> Metric:
        """Mean across all instances and seeds for a single perturbation level."""
        df = self._rob
        sub = df[(df['perturb_type'] == perturb_type) &
                 (df['level'].astype(float).round(6) == round(float(level), 6)) &
                 (df['method'] == method.lower())]
        if sub.empty:
            raise KeyError(f'no log entry for {(perturb_type, level, method)}')
        return Metric(completion=float(sub['completion'].mean()),
                      tardiness=float(sub['tardy'].mean()))

    # ----- ablation lookups -----
    def ablation_component(self, banks: int, window: str, variant: str,
                           seed: int, pattern: str = 'Uniform',
                           size: int = 200, twotc: str = '10%') -> Metric:
        df = self._abl
        row = df[(df['banks'] == banks) &
                 (df['window'] == window) &
                 (df['variant'] == variant) &
                 (df['pattern'] == pattern) &
                 (df['size'] == size) &
                 (df['twotc'] == twotc) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for {(banks, window, variant, seed)}')
        return Metric(completion=float(row.iloc[0]['completion']), tardiness=0.0)

    def ablation_tlo(self, variant: str, seed: int,
                     pattern: str = 'Uniform', size: int = 200,
                     twotc: str = '10%') -> Metric:
        df = self._tlo
        row = df[(df['variant'] == variant) &
                 (df['pattern'] == pattern) &
                 (df['size'] == size) &
                 (df['twotc'] == twotc) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for TLO ({variant}, seed={seed})')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']))

    def reward_scale(self, scale: float, method: str, seed: int) -> Metric:
        df = self._rs
        row = df[(df['scale'].astype(float).round(6) == round(float(scale), 6)) &
                 (df['method'] == method) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for reward-scale ({scale}, {method}, seed={seed})')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']))

    # ----- MORL lookup -----
    def morl(self, method: str, seed: int,
             pattern: str = 'Uniform', size: int = 200,
             twotc: str = '10%') -> Metric:
        df = self._morl
        row = df[(df['method'] == method) &
                 (df['pattern'] == pattern) &
                 (df['size'] == size) &
                 (df['twotc'] == twotc) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for MORL ({method}, seed={seed})')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']))

    # ----- architecture lookup -----
    def architecture(self, method: str, seed: int,
                     pattern: str = 'Uniform', size: int = 200,
                     twotc: str = '10%') -> Metric:
        df = self._arch
        row = df[(df['method'] == method) &
                 (df['pattern'] == pattern) &
                 (df['size'] == size) &
                 (df['twotc'] == twotc) &
                 (df['seed'] == seed)]
        if row.empty:
            raise KeyError(f'no log entry for architecture ({method}, seed={seed})')
        r = row.iloc[0]
        return Metric(completion=float(r['completion']),
                      tardiness=float(r['tardy']))


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

# Older scripts referenced `BenchmarkAggregator`; the class was renamed
# when the per-seed log structure superseded the previous in-memory
# aggregate model. The alias preserves backwards compatibility.
BenchmarkAggregator = ExperimentLog
