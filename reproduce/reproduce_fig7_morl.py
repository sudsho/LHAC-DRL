"""Figure 7 -- MORL Pareto front (LHAC vs constrained / lexicographic /
Pareto-front baselines, with and without the DAF augmentations).

  python reproduce/reproduce_fig7_morl.py \
         --data data/industrial_servers_2024.csv --seeds 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if THIS not in sys.path:
    sys.path.insert(0, THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _common import RESULTS, COLOR_LHAC, LINE_PALETTE, out
from _pipeline import morl_sweep

LABELS = {
    'lhac_daf':            'LHAC (full DAF)',
    'lppo':                'LPPO (bare)',
    'lppo_daf':            'LPPO + DAF',
    'ppo_lagrangian':      'PPO-Lagrangian (bare)',
    'ppo_lagrangian_daf':  'PPO-Lag + DAF',
    'rcpo':                'RCPO (bare)',
    'rcpo_daf':            'RCPO + DAF',
    'envelope_morl':       'Envelope MORL (bare)',
    'envelope_morl_daf':   'Envelope MORL + DAF',
    'weighted_rl':         'Weighted RL (bare)',
    'weighted_rl_daf':     'Weighted RL + DAF',
}

PALETTE = {
    'lhac_daf':            COLOR_LHAC,
    'lppo':                LINE_PALETTE['lppo'],
    'lppo_daf':            LINE_PALETTE['lppo'],
    'ppo_lagrangian':      LINE_PALETTE['ppo_lagrangian'],
    'ppo_lagrangian_daf':  LINE_PALETTE['ppo_lagrangian'],
    'rcpo':                LINE_PALETTE['rcpo'],
    'rcpo_daf':            LINE_PALETTE['rcpo'],
    'envelope_morl':       LINE_PALETTE['envelope_morl'],
    'envelope_morl_daf':   LINE_PALETTE['envelope_morl'],
    'weighted_rl':         LINE_PALETTE['weighted_rl'],
    'weighted_rl_daf':     LINE_PALETTE['weighted_rl'],
}

MARKERS_BARE = {'lppo': 's', 'ppo_lagrangian': 'D', 'rcpo': 'v',
                'envelope_morl': '^', 'weighted_rl': 'X'}


def _plot(df: pd.DataFrame, out_path: str) -> None:
    df = df.set_index('method')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')

    if 'lhac_daf' in df.index:
        r = df.loc['lhac_daf']
        ax.errorbar(r['tardy'], r['completion'],
                    xerr=r['tardy_sd'], yerr=r['completion_sd'],
                    fmt='*', ms=22, color=COLOR_LHAC,
                    markeredgecolor='black', markeredgewidth=1.5,
                    elinewidth=1.5, capsize=5,
                    label=LABELS['lhac_daf'], zorder=10)

    for k in MARKERS_BARE:
        if k in df.index:
            r = df.loc[k]
            ax.errorbar(r['tardy'], r['completion'],
                        xerr=r['tardy_sd'], yerr=r['completion_sd'],
                        fmt=MARKERS_BARE[k], ms=14,
                        markerfacecolor='none', markeredgecolor=PALETTE[k],
                        markeredgewidth=2.0,
                        ecolor=PALETTE[k], elinewidth=1.2, capsize=4,
                        label=LABELS[k])
        d = f'{k}_daf'
        if d in df.index:
            r = df.loc[d]
            ax.errorbar(r['tardy'], r['completion'],
                        xerr=r['tardy_sd'], yerr=r['completion_sd'],
                        fmt=MARKERS_BARE[k], ms=14, color=PALETTE[d],
                        markeredgecolor='black', markeredgewidth=1.0,
                        ecolor=PALETTE[d], elinewidth=1.2, capsize=4,
                        label=LABELS[d])

    ax.set_xlabel('Tardiness rate (%)', fontsize=14)
    ax.set_ylabel('Completion rate (%)', fontsize=14)
    ax.set_title('MORL Pareto comparison -- completion-first lexicographic priority',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/industrial_servers_2024.csv')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--use-cache', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = out('Figure_07_MORL_Pareto.png')
    csv_path = os.path.join(RESULTS, 'morl_baselines.csv')
    if not args.use_cache:
        if os.path.exists(args.data):
            print(f'[load] using confidential dataset: {args.data}')
        else:
            print(f'[load] {args.data} not present; using calibrated synthetic'
                  f' instances')
        print(f'[sweep] 11 MORL methods x 27 instances x {args.seeds} seeds')
        t0 = time.time()
        morl_sweep(seeds=args.seeds, out_csv=csv_path)
        print(f'\n[done] MORL sweep completed in {(time.time() - t0)/60:.1f} min')
    else:
        print('[cache] reading aggregated MORL statistics...')
    df = pd.read_csv(csv_path)
    _plot(df, out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
