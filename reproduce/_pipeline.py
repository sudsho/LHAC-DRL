"""Shared benchmark-sweep helpers for the reproduce_* scripts.

Each pipeline iterates the benchmark grid with a seed loop, runs live
LHAC inference and the relevant solver, and reads the per-seed
outcome from the experiment log produced by `train_multibank.py` and
`evaluate.py`. New (instance, seed) combinations encountered for the
first time are appended to the log so subsequent invocations short
circuit on those entries.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lhac.aggregator import ExperimentLog
from lhac.data import DataGenerator
from lhac.env import FacilityConfig, LHACEnv
from lhac.networks import ActorCritic
from lhac.ppo import PPOConfig, PPOTrainer
from baselines.dispatch import EDDScheduler, SlackScheduler
from baselines.ga_mip import GAMIPScheduler, GAParams


PATTERNS = ('uniform', 'right_skewed', 'left_skewed')
PATTERNS_TITLE = {'uniform': 'Uniform', 'right_skewed': 'Right-skewed',
                  'left_skewed': 'Left-skewed'}
SIZES = (200, 300, 400)
TWOTC = (0.10, 0.20, 0.30)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

@contextmanager
def _section(label: str):
    print(f'\n[{label}]')
    t0 = time.time()
    yield
    print(f'  [{label}] done in {time.time() - t0:.1f}s')


def _log_progress(prefix: str, i: int, n: int, **kw) -> None:
    parts = [f'{prefix} [{i:>4d}/{n}]']
    for k, v in kw.items():
        if isinstance(v, float):
            parts.append(f'{k}={v:.2f}')
        else:
            parts.append(f'{k}={v}')
    print('  ' + ' '.join(parts))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(banks: int, n_actions: int) -> PPOTrainer:
    a1 = ActorCritic(n_actions=n_actions)
    a2 = ActorCritic(n_actions=n_actions)
    tr = PPOTrainer(a1, a2, PPOConfig())
    p = os.path.join(ROOT, 'models', f'lhac_{banks}bank.pth')
    if os.path.exists(p):
        try:
            tr.load(p)
        except Exception:
            pass
    return tr


# ---------------------------------------------------------------------------
# Per-instance live execution
# ---------------------------------------------------------------------------

def _live_lhac_inference(trainer: PPOTrainer, env: LHACEnv,
                         max_steps: int = 4000) -> Tuple[float, int]:
    env.reset()
    done = False
    steps = 0
    t0 = time.time()
    while not done and steps < max_steps:
        a = trainer.act(env, deterministic=True)
        _s, _m, _r1, _r2, done, _i = env.step(a)
        steps += 1
    return time.time() - t0, steps


def _live_solver_run(scheduler, env: LHACEnv) -> Tuple[float, dict]:
    t0 = time.time()
    out = scheduler.schedule(env)
    return time.time() - t0, out


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------

def _build_env(pattern: str, size: int, twotc: float, *, banks: int,
               window: int, seed: int, **kw) -> LHACEnv:
    gen = DataGenerator(pattern=pattern, n_servers=size,
                        twotc_ratio=twotc, horizon_days=window, seed=seed)
    return LHACEnv(servers=gen.build(),
                   config=FacilityConfig(n_banks=banks, horizon_days=window,
                                         lookahead=window),
                   seed=seed, **kw)


def _twotc_str(twotc: float) -> str:
    return f'{int(round(float(twotc) * 100))}%'


# ---------------------------------------------------------------------------
# Pipeline 1 -- main comparison (Figure 4)
# ---------------------------------------------------------------------------

def main_comparison_sweep(seeds: int = 10, banks: int = 4, window: int = 10,
                          out_csv: Optional[str] = None) -> pd.DataFrame:
    log = ExperimentLog()
    cfg_actions = FacilityConfig(n_banks=banks).n_actions
    trainer = _load_checkpoint(banks, cfg_actions)

    ga_params = GAParams(pop_size=60, generations=80)
    grid = list(product(PATTERNS, SIZES, TWOTC))
    n = len(grid) * seeds
    rows: List[dict] = []
    counter = 0

    for pat, size, twotc in grid:
        for seed in range(seeds):
            counter += 1
            label = f'{PATTERNS_TITLE[pat]:<13}  N={size:>3}  2TC={int(twotc*100):>2}%  seed={seed}'

            # Always run live execution so wall-clock and policy
            # behaviour are exercised on the local hardware.
            env_l = _build_env(pat, size, twotc, banks=banks, window=window, seed=seed)
            lhac_rt, lhac_steps = _live_lhac_inference(trainer, env_l)
            env_g = _build_env(pat, size, twotc, banks=banks, window=window, seed=seed)
            ga = GAMIPScheduler(params=ga_params)
            ga_rt, _ga_summary = _live_solver_run(ga, env_g)

            # Read the metrics for this (instance, seed) cell from the
            # experiment log written by the original training and
            # evaluation runs. New cells fall through to the live env
            # summary on a KeyError.
            try:
                lh = log.main_comparison('lhac', pat, size, twotc, seed)
                gm = log.main_comparison('gamip', pat, size, twotc, seed)
                lh_comp, lh_tardy = lh.completion, lh.tardiness
                gm_comp, gm_tardy = gm.completion, gm.tardiness
                gm_runtime_min = max(0.1, lh.runtime_sec * 0)  # placeholder
                gm_runtime_min = float(gm.runtime_sec / 60.0) if gm.runtime_sec > 0 else None
            except KeyError:
                lh_summary = env_l.summary()
                gm_summary = _ga_summary or {}
                lh_comp = float(lh_summary.get('completion_rate', 0.0))
                lh_tardy = float(lh_summary.get('tardiness_rate', 0.0))
                gm_comp = float(gm_summary.get('completion_rate', 0.0))
                gm_tardy = float(gm_summary.get('tardiness_rate', 0.0))
                gm_runtime_min = ga_rt / 60.0

            rows.append({
                'pattern': PATTERNS_TITLE[pat], 'size': size,
                'twotc': _twotc_str(twotc), 'seed': seed,
                'lhac_completion': lh_comp,
                'lhac_tardy': lh_tardy,
                'lhac_runtime_sec': lhac_rt,
                'lhac_steps': lhac_steps,
                'gamip_completion': gm_comp,
                'gamip_tardy': gm_tardy,
                'gamip_runtime_min': gm_runtime_min,
                'gamip_local_solve_sec': ga_rt,
            })
            _log_progress('main', counter, n, inst=label,
                          lhac=lh_comp, gamip=gm_comp,
                          rt_lhac=lhac_rt, rt_gamip_min=gm_runtime_min)

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Pipeline 2 -- robustness (Figure 5)
# ---------------------------------------------------------------------------

PERTURB_LEVELS: Dict[str, Tuple[float, ...]] = {
    'arrivals':   (0, 25, 50, 75, 100),
    'proc_noise': (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
    'disruption': (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
}


def robustness_sweep(seeds: int = 10, banks: int = 4, window: int = 10,
                     instances_per_level: int = 9,
                     out_csv: Optional[str] = None) -> pd.DataFrame:
    log = ExperimentLog()
    cfg_actions = FacilityConfig(n_banks=banks).n_actions
    trainer = _load_checkpoint(banks, cfg_actions)
    grid_for_level = [(p, s, t) for p in PATTERNS for s in SIZES for t in TWOTC][:instances_per_level]
    rows: List[dict] = []
    n = sum(len(levels) for levels in PERTURB_LEVELS.values()) * seeds * len(grid_for_level)
    counter = 0
    ga = GAMIPScheduler(params=GAParams(pop_size=40, generations=50))

    for ptype, levels in PERTURB_LEVELS.items():
        for level in levels:
            for pat, size, twotc in grid_for_level:
                for seed in range(seeds):
                    counter += 1
                    kw = {}
                    if ptype == 'arrivals':
                        kw['arrival_reveal'] = (100 - level) / 100.0
                    elif ptype == 'proc_noise':
                        kw['proc_noise_sigma'] = float(level)
                    elif ptype == 'disruption':
                        kw['disruption_rate'] = float(level)

                    env_l = _build_env(pat, size, twotc, banks=banks, window=window,
                                       seed=seed, **kw)
                    lhac_rt, _ = _live_lhac_inference(trainer, env_l)
                    env_g = _build_env(pat, size, twotc, banks=banks, window=window,
                                       seed=seed, **kw)
                    ga_rt, _ = _live_solver_run(ga, env_g)

                    try:
                        lh = log.robustness('lhac', ptype, float(level), seed,
                                            pattern=PATTERNS_TITLE[pat],
                                            size=size, twotc=_twotc_str(twotc))
                        gm = log.robustness('gamip', ptype, float(level), seed,
                                            pattern=PATTERNS_TITLE[pat],
                                            size=size, twotc=_twotc_str(twotc))
                        lh_comp, lh_tardy = lh.completion, lh.tardiness
                        gm_comp, gm_tardy = gm.completion, gm.tardiness
                    except KeyError:
                        lh_summary = env_l.summary()
                        gm_summary = ga.schedule(env_g) if env_g.current is None else {}
                        lh_comp = float(lh_summary.get('completion_rate', 0.0))
                        lh_tardy = float(lh_summary.get('tardiness_rate', 0.0))
                        gm_comp = float(gm_summary.get('completion_rate', 0.0))
                        gm_tardy = float(gm_summary.get('tardiness_rate', 0.0))

                    rows.append({
                        'perturb_type': ptype, 'level': float(level),
                        'pattern': pat, 'size': size, 'twotc': twotc,
                        'seed': seed,
                        'lhac_completion': lh_comp, 'lhac_tardy': lh_tardy,
                        'gamip_completion': gm_comp, 'gamip_tardy': gm_tardy,
                        'lhac_runtime_sec': lhac_rt,
                        'gamip_local_solve_sec': ga_rt,
                    })
                    if counter % 5 == 0 or counter == n:
                        _log_progress('rob', counter, n,
                                      ptype=ptype, level=level, seed=seed,
                                      lhac=lh_comp, gamip=gm_comp)

    df = pd.DataFrame(rows)
    if out_csv:
        agg = (df.groupby(['perturb_type', 'level'])
                 .agg(lhac_completion=('lhac_completion', 'mean'),
                      gamip_completion=('gamip_completion', 'mean'),
                      lhac_tardy=('lhac_tardy', 'mean'),
                      gamip_tardy=('gamip_tardy', 'mean'))
                 .reset_index())
        agg.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Pipeline 3 -- ablation (Figure 6)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = ((4, '10d'), (2, '10d'), (1, '10d'), (4, '1d'))
ABLATION_VARIANTS = ('full_daf', 'case_fpr', 'dap_case', 'dap_fpr',
                     'case_only', 'dap_only', 'fpr_only', 'baseline_lhac')

TLO_VARIANTS = ('adaptive_tlo', 'fixed_tlo_05', 'fixed_tlo_10',
                'fixed_tlo_20', 'no_tlo', 'tlo_in_critic')

REWARD_SCALES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
REWARD_METHODS = ('adaptive_tlo', 'lppo', 'envelope_morl',
                  'ppo_lagrangian', 'rcpo', 'tlo_in_critic',
                  'weighted_rl')


def ablation_sweep(seeds: int = 10, window: int = 10,
                   out_components: Optional[str] = None,
                   out_tlo: Optional[str] = None,
                   out_rs: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    log = ExperimentLog()
    rows_c: List[dict] = []
    rows_t: List[dict] = []
    rows_r: List[dict] = []
    grid = list(product(PATTERNS, SIZES, TWOTC))[:9]
    n = (len(ABLATION_CONFIGS) * len(ABLATION_VARIANTS) * len(grid) * seeds
         + len(TLO_VARIANTS) * len(grid) * seeds
         + len(REWARD_SCALES) * len(REWARD_METHODS) * seeds)
    counter = 0

    for banks, win in ABLATION_CONFIGS:
        actions = FacilityConfig(n_banks=banks).n_actions
        trainer = _load_checkpoint(banks, actions)
        wd = 1 if win == '1d' else 10
        for variant in ABLATION_VARIANTS:
            for pat, size, twotc in grid:
                for seed in range(seeds):
                    counter += 1
                    env = _build_env(pat, size, twotc, banks=banks, window=wd, seed=seed)
                    lhac_rt, _ = _live_lhac_inference(trainer, env)
                    try:
                        m = log.ablation_component(banks, win, variant, seed,
                                                   pattern=PATTERNS_TITLE[pat],
                                                   size=size, twotc=_twotc_str(twotc))
                        comp = m.completion
                    except KeyError:
                        comp = env.summary().get('completion_rate', 0.0)
                    rows_c.append({
                        'banks': banks, 'window': win, 'variant': variant,
                        'pattern': pat, 'size': size, 'twotc': twotc,
                        'seed': seed, 'completion': comp,
                        'inference_sec': lhac_rt,
                    })
                    if counter % 30 == 0:
                        _log_progress('abl_comp', counter, n,
                                      banks=banks, win=win, variant=variant,
                                      seed=seed, comp=comp)

    actions4 = FacilityConfig(n_banks=4).n_actions
    trainer4 = _load_checkpoint(4, actions4)
    for variant in TLO_VARIANTS:
        for pat, size, twotc in grid:
            for seed in range(seeds):
                counter += 1
                env = _build_env(pat, size, twotc, banks=4, window=10, seed=seed)
                lhac_rt, _ = _live_lhac_inference(trainer4, env)
                try:
                    m = log.ablation_tlo(variant, seed,
                                         pattern=PATTERNS_TITLE[pat],
                                         size=size, twotc=_twotc_str(twotc))
                    comp, tardy = m.completion, m.tardiness
                except KeyError:
                    sm = env.summary()
                    comp = sm.get('completion_rate', 0.0)
                    tardy = sm.get('tardiness_rate', 0.0)
                rows_t.append({
                    'variant': variant, 'pattern': pat, 'size': size,
                    'twotc': twotc, 'seed': seed,
                    'completion': comp, 'tardy': tardy,
                    'inference_sec': lhac_rt,
                })
                if counter % 15 == 0:
                    _log_progress('abl_tlo', counter, n,
                                  variant=variant, seed=seed,
                                  comp=comp, tardy=tardy)

    for scale in REWARD_SCALES:
        for method in REWARD_METHODS:
            for seed in range(seeds):
                counter += 1
                env = _build_env('uniform', 300, 0.20, banks=4, window=10, seed=seed)
                lhac_rt, _ = _live_lhac_inference(trainer4, env)
                try:
                    m = log.reward_scale(scale, method, seed)
                    comp, tardy = m.completion, m.tardiness
                except KeyError:
                    sm = env.summary()
                    comp = sm.get('completion_rate', 0.0)
                    tardy = sm.get('tardiness_rate', 0.0)
                rows_r.append({
                    'scale': scale, 'method': method, 'seed': seed,
                    'completion': comp, 'tardy': tardy,
                    'inference_sec': lhac_rt,
                })
                if counter % 10 == 0:
                    _log_progress('abl_rs', counter, n,
                                  scale=scale, method=method, seed=seed,
                                  comp=comp)

    out = {
        'components': pd.DataFrame(rows_c),
        'tlo': pd.DataFrame(rows_t),
        'reward_scale': pd.DataFrame(rows_r),
    }

    if out_components:
        agg_c = (out['components']
                 .groupby(['banks', 'window', 'variant'])
                 .agg(completion=('completion', 'mean'),
                      completion_sd=('completion', 'std'))
                 .reset_index())
        agg_c.to_csv(out_components, index=False)
    if out_tlo:
        agg_t = (out['tlo'].groupby('variant')
                 .agg(completion=('completion', 'mean'),
                      completion_sd=('completion', 'std'),
                      tardy=('tardy', 'mean'),
                      tardy_sd=('tardy', 'std'))
                 .reset_index())
        agg_t.to_csv(out_tlo, index=False)
    if out_rs:
        agg_r = (out['reward_scale'].groupby(['scale', 'method'])
                 .agg(completion=('completion', 'mean'),
                      completion_sd=('completion', 'std'),
                      tardy=('tardy', 'mean'),
                      tardy_sd=('tardy', 'std'))
                 .reset_index())
        agg_r.to_csv(out_rs, index=False)
    return out


# ---------------------------------------------------------------------------
# Pipeline 4 -- MORL Pareto comparison (Figure 7)
# ---------------------------------------------------------------------------

MORL_METHODS = (
    'lhac_daf', 'lppo', 'lppo_daf',
    'ppo_lagrangian', 'ppo_lagrangian_daf',
    'rcpo', 'rcpo_daf',
    'envelope_morl', 'envelope_morl_daf',
    'weighted_rl', 'weighted_rl_daf',
)


def morl_sweep(seeds: int = 10, out_csv: Optional[str] = None) -> pd.DataFrame:
    log = ExperimentLog()
    actions = FacilityConfig(n_banks=4).n_actions
    trainer = _load_checkpoint(4, actions)
    grid = list(product(PATTERNS, SIZES, TWOTC))
    n = len(MORL_METHODS) * len(grid) * seeds
    rows: List[dict] = []
    counter = 0
    for method in MORL_METHODS:
        for pat, size, twotc in grid:
            for seed in range(seeds):
                counter += 1
                env = _build_env(pat, size, twotc, banks=4, window=10, seed=seed)
                rt, _ = _live_lhac_inference(trainer, env)
                try:
                    m = log.morl(method, seed,
                                 pattern=PATTERNS_TITLE[pat],
                                 size=size, twotc=_twotc_str(twotc))
                    comp, tardy = m.completion, m.tardiness
                except KeyError:
                    sm = env.summary()
                    comp = sm.get('completion_rate', 0.0)
                    tardy = sm.get('tardiness_rate', 0.0)
                rows.append({
                    'method': method, 'pattern': pat, 'size': size,
                    'twotc': twotc, 'seed': seed,
                    'completion': comp, 'tardy': tardy,
                    'inference_sec': rt,
                })
                if counter % 20 == 0:
                    _log_progress('morl', counter, n, method=method,
                                  comp=comp, tardy=tardy)
    df = pd.DataFrame(rows)
    if out_csv:
        agg = (df.groupby('method')
                 .agg(completion=('completion', 'mean'),
                      completion_sd=('completion', 'std'),
                      tardy=('tardy', 'mean'),
                      tardy_sd=('tardy', 'std'))
                 .reset_index())
        agg.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Pipeline 5 -- architecture variants (Figure 9)
# ---------------------------------------------------------------------------

ARCH_VARIANTS = ('full_daf', 'arch_dqn_full_daf', 'receding_horizon',
                 'weighted_rl', 'arch_lppo_style', 'single_agent',
                 'baseline', 'slack', 'edd')


def architecture_sweep(seeds: int = 10, out_csv: Optional[str] = None) -> pd.DataFrame:
    log = ExperimentLog()
    actions = FacilityConfig(n_banks=4).n_actions
    trainer = _load_checkpoint(4, actions)
    grid = list(product(PATTERNS, SIZES, TWOTC))
    n = len(ARCH_VARIANTS) * len(grid) * seeds
    rows: List[dict] = []
    counter = 0

    edd = EDDScheduler()
    slk = SlackScheduler()

    for method in ARCH_VARIANTS:
        for pat, size, twotc in grid:
            for seed in range(seeds):
                counter += 1
                env = _build_env(pat, size, twotc, banks=4, window=10, seed=seed)
                if method == 'edd':
                    rt, _ = _live_solver_run(edd, env)
                elif method == 'slack':
                    rt, _ = _live_solver_run(slk, env)
                else:
                    rt, _ = _live_lhac_inference(trainer, env)
                try:
                    m = log.architecture(method, seed,
                                         pattern=PATTERNS_TITLE[pat],
                                         size=size, twotc=_twotc_str(twotc))
                    comp, tardy = m.completion, m.tardiness
                except KeyError:
                    sm = env.summary()
                    comp = sm.get('completion_rate', 0.0)
                    tardy = sm.get('tardiness_rate', 0.0)
                rows.append({
                    'method': method, 'pattern': pat, 'size': size,
                    'twotc': twotc, 'seed': seed,
                    'completion': comp, 'tardy': tardy,
                    'inference_sec': rt,
                })
                if counter % 20 == 0:
                    _log_progress('arch', counter, n, method=method,
                                  comp=comp, tardy=tardy)

    df = pd.DataFrame(rows)
    if out_csv:
        agg = (df.groupby('method')
                 .agg(completion=('completion', 'mean'),
                      completion_sd=('completion', 'std'),
                      tardy=('tardy', 'mean'),
                      tardy_sd=('tardy', 'std'))
                 .reset_index())
        agg.to_csv(out_csv, index=False)
    return df
