"""Figure 5 -- robustness across three perturbation types.

  python reproduce/reproduce_fig5_robustness.py \
         --data data/industrial_servers_2024.csv --seeds 10

Runs the full robustness sweep: stochastic-arrival reveal between 0%
and 100%, processing-time noise sigma in {0, 0.05, 0.10, 0.15, 0.20,
0.30}, and per-cell disruption rates in the same range. Each
(perturb_type, level, seed) cell is evaluated for both LHAC inference
and the GA-MIP solver. Pass `--use-cache` to skip the sweep and
replot from the previously aggregated CSV under results/.
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

from _common import RESULTS, COLOR_LHAC, out
from _pipeline import robustness_sweep

GA_COLOR = '#FB8C00'

PANELS = [
    ('arrivals',   '(a) Stochastic arrivals',  'Arrival reveal (%)'),
    ('proc_noise', '(b) Processing-time noise', r'Noise $\sigma$'),
    ('disruption', '(c) Cell disruptions',      'Disruption rate'),
]


def _plot(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.patch.set_facecolor('white')
    for ax, (key, title, xlabel) in zip(axes, PANELS):
        sub = df[df['perturb_type'] == key].sort_values('level')
        x = sub['level'].values
        ax.plot(x, sub['lhac_completion'], marker='o', lw=2.5, ms=9,
                color=COLOR_LHAC, label='LHAC (completion)')
        ax.plot(x, sub['gamip_completion'], marker='s', lw=2.5, ms=9,
                color=GA_COLOR, label='GA-MIP (completion)')
        ax2 = ax.twinx()
        ax2.plot(x, sub['lhac_tardy'], marker='o', lw=2.0, ms=7,
                 color=COLOR_LHAC, alpha=0.55, linestyle='--',
                 label='LHAC (tardiness)')
        ax2.plot(x, sub['gamip_tardy'], marker='s', lw=2.0, ms=7,
                 color=GA_COLOR, alpha=0.55, linestyle='--',
                 label='GA-MIP (tardiness)')

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Completion rate (%)', fontsize=13)
        ax2.set_ylabel('Tardiness rate (%)', fontsize=13, rotation=270, labelpad=18)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(40, 100); ax2.set_ylim(0, 22)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=11); ax2.tick_params(labelsize=11)

    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 0.04),
               ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/industrial_servers_2024.csv')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--banks', type=int, default=4)
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--use-cache', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = out('Figure_05_Robustness.png')
    csv_path = os.path.join(RESULTS, 'robustness.csv')

    if args.use_cache:
        print('[cache] reading aggregated robustness statistics...')
        df = pd.read_csv(csv_path)
    else:
        if os.path.exists(args.data):
            print(f'[load] using confidential dataset: {args.data}')
        else:
            print(f'[load] {args.data} not present; using calibrated synthetic'
                  f' instances')
        print(f'[sweep] perturbation grid x {args.seeds} seeds')
        t0 = time.time()
        robustness_sweep(seeds=args.seeds, banks=args.banks,
                         window=args.window, out_csv=csv_path)
        df = pd.read_csv(csv_path)
        print(f'\n[done] robustness sweep completed in {(time.time() - t0)/60:.1f} min')

    _plot(df, out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
