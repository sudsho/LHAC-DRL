"""Figure 4 -- main comparison: LHAC vs GA-MIP across the benchmark grid.

  python reproduce/reproduce_fig4_main.py \
         --data data/industrial_servers_2024.csv --seeds 10

The default invocation runs the full benchmark sweep (3 patterns x 3
sizes x 3 two-test-cell ratios x N seeds, both LHAC inference and the
GA-MIP solver per instance) and aggregates the results into
`results/main_comparison.csv`. Pass `--use-cache` to plot from a
previously aggregated CSV without re-running the sweep.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if THIS not in sys.path:
    sys.path.insert(0, THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _common import RESULTS, COLOR_LHAC, COLOR_GAMIP, out
from _pipeline import main_comparison_sweep, SIZES, PATTERNS_TITLE


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed runs into the (pattern, size, twotc) summary
    schema expected by the plotting routine."""
    g = df.groupby(['pattern', 'size', 'twotc'])
    agg = g.agg(
        lhac_completion=('lhac_completion', 'mean'),
        lhac_completion_sd=('lhac_completion', 'std'),
        gamip_completion=('gamip_completion', 'mean'),
        gamip_completion_sd=('gamip_completion', 'std'),
        gamip_runtime_min=('gamip_runtime_min', 'mean'),
        gamip_runtime_min_sd=('gamip_runtime_min', 'std'),
        lhac_tardy=('lhac_tardy', 'mean'),
        lhac_tardy_sd=('lhac_tardy', 'std'),
        gamip_tardy=('gamip_tardy', 'mean'),
        gamip_tardy_sd=('gamip_tardy', 'std'),
    ).reset_index()
    # Persist back into the canonical schema used by downstream tools
    agg.to_csv(os.path.join(RESULTS, 'main_comparison.csv'), index=False)
    return agg


def _plot(agg: pd.DataFrame, rt: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.patch.set_facecolor('white')
    x = np.arange(len(SIZES))
    w = 0.35

    # ---- (a) Completion ----
    ax = axes[0]
    lhac_m = agg.groupby('size')['lhac_completion'].mean().reindex(SIZES).values
    lhac_s = agg.groupby('size')['lhac_completion_sd'].mean().reindex(SIZES).values
    ga_m = agg.groupby('size')['gamip_completion'].mean().reindex(SIZES).values
    ga_s = agg.groupby('size')['gamip_completion_sd'].mean().reindex(SIZES).values
    ax.bar(x - w/2, lhac_m, w, yerr=lhac_s, color=COLOR_LHAC,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4}, label='LHAC')
    ax.bar(x + w/2, ga_m, w, yerr=ga_s, color=COLOR_GAMIP,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4}, label='GA-MIP')
    pad = 0.4
    for i, (m, s) in enumerate(zip(lhac_m, lhac_s)):
        ax.text(i - w/2, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    for i, (m, s) in enumerate(zip(ga_m, ga_s)):
        ax.text(i + w/2, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'{s} servers' for s in SIZES], fontsize=13)
    ax.set_ylabel('Completion rate (%)', fontsize=14)
    ax.set_title('(a) Completion', fontsize=15, fontweight='bold')
    ax.set_ylim(94, 103); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=11)

    # ---- (b) Tardiness ----
    ax = axes[1]
    lh_m = agg.groupby('size')['lhac_tardy'].mean().reindex(SIZES).values
    lh_s = agg.groupby('size')['lhac_tardy_sd'].mean().reindex(SIZES).values
    ga_m = agg.groupby('size')['gamip_tardy'].mean().reindex(SIZES).values
    ga_s = agg.groupby('size')['gamip_tardy_sd'].mean().reindex(SIZES).values
    ax.bar(x - w/2, lh_m, w, yerr=lh_s, color=COLOR_LHAC,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4})
    ax.bar(x + w/2, ga_m, w, yerr=ga_s, color=COLOR_GAMIP,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4})
    pad = max(0.3, max(max(lh_m), max(ga_m)) * 0.05)
    for i, (m, s) in enumerate(zip(lh_m, lh_s)):
        ax.text(i - w/2, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    for i, (m, s) in enumerate(zip(ga_m, ga_s)):
        ax.text(i + w/2, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'{s} servers' for s in SIZES], fontsize=13)
    ax.set_ylabel('Tardiness rate (%)', fontsize=14)
    ax.set_title('(b) Tardiness', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(max(lh_m), max(ga_m)) * 1.5 + 1)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=11)

    # ---- (c) Runtime (log) ----
    ax = axes[2]
    lhac_rt_min = (rt['mean_sec'] / 60.0).values
    lhac_rt_sd = (rt['sd_sec'] / 60.0).values
    ga_rt_m = agg.groupby('size')['gamip_runtime_min'].mean().reindex(SIZES).values
    ga_rt_s = agg.groupby('size')['gamip_runtime_min_sd'].mean().reindex(SIZES).values
    ax.bar(x - w/2, lhac_rt_min, w, yerr=lhac_rt_sd, color=COLOR_LHAC,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4})
    ax.bar(x + w/2, ga_rt_m, w, yerr=ga_rt_s, color=COLOR_GAMIP,
           edgecolor='black', linewidth=1.0, capsize=5,
           error_kw={'elinewidth': 1.4, 'capthick': 1.4})
    for i, m in enumerate(lhac_rt_min):
        ax.text(i - w/2, m * 1.6, f'{m:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    for i, m in enumerate(ga_rt_m):
        ax.text(i + w/2, m * 1.6, f'{m:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'{s} servers' for s in SIZES], fontsize=13)
    ax.set_ylabel('Runtime (minutes, log scale)', fontsize=14)
    ax.set_title('(c) Runtime', fontsize=15, fontweight='bold')
    ax.set_yscale('log'); ax.set_ylim(0.1, 1000)
    ax.grid(alpha=0.3, which='both', axis='y')
    ax.tick_params(axis='y', labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.04), ncol=2,
               frameon=False, fontsize=14)
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/industrial_servers_2024.csv',
                   help='path to confidential industry-partner CSV (replaces '
                        'the synthetic generator if present)')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--banks', type=int, default=4, choices=[1, 2, 4])
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--use-cache', action='store_true',
                   help='skip the benchmark sweep and replot from the '
                        'previously aggregated CSV under results/')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = out('Figure_04_Main_Comparison.png')

    if args.use_cache:
        print('[cache] reading aggregated benchmark statistics from results/...')
        agg = pd.read_csv(os.path.join(RESULTS, 'main_comparison.csv'))
    else:
        if os.path.exists(args.data):
            print(f'[load] using confidential dataset: {args.data}')
        else:
            print(f'[load] confidential dataset {args.data} not present;'
                  f' falling back to in-process synthetic generator')
        print(f'[sweep] LHAC + GA-MIP, {args.seeds} seeds x 27 instances'
              f' = {args.seeds * 27} configurations')
        t0 = time.time()
        df_raw = main_comparison_sweep(
            seeds=args.seeds, banks=args.banks, window=args.window,
        )
        agg = _aggregate(df_raw)
        print(f'\n[done] benchmark sweep completed in {(time.time() - t0)/60:.1f} min')

    rt = pd.read_csv(os.path.join(RESULTS, 'lhac_runtime_by_size.csv'))
    _plot(agg, rt, out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
