"""Figure 9 -- architecture variant comparison.

  python reproduce/reproduce_fig9_arch.py

LHAC vs DQN+DAF, Receding-Horizon CPLEX, Weighted RL, LPPO-style,
Single Agent, Baseline LHAC, EDD, Slack.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if THIS not in sys.path:
    sys.path.insert(0, THIS)

from _common import (RESULTS, COLOR_LHAC, COLOR_GRAY_DARK,
                     COLOR_GRAY_MED, COLOR_GRAY_LIGHT,
                     COLOR_GRAY_FAINT, out)

ORDER = [
    ('full_daf',           'LHAC',           COLOR_LHAC),
    ('arch_dqn_full_daf',  'DQN+DAF',        COLOR_GRAY_DARK),
    ('receding_horizon',   'Receding-\nHorizon', COLOR_GRAY_MED),
    ('weighted_rl',        'Weighted RL',    COLOR_GRAY_LIGHT),
    ('arch_lppo_style',    'LPPO-style',     COLOR_GRAY_LIGHT),
    ('single_agent',       'Single Agent',   COLOR_GRAY_LIGHT),
    ('baseline',           'Baseline LHAC',  COLOR_GRAY_FAINT),
    ('slack',              'Slack',          COLOR_GRAY_FAINT),
    ('edd',                'EDD',            COLOR_GRAY_FAINT),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cache', action='store_true')
    parser.parse_args()

    df = pd.read_csv(os.path.join(RESULTS, 'architecture_variants.csv')).set_index('method')

    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor('white')

    labels = [a[1] for a in ORDER]
    comps = [df.loc[k, 'completion'] for k, _, _ in ORDER]
    csd   = [df.loc[k, 'completion_sd'] for k, _, _ in ORDER]
    tars  = [df.loc[k, 'tardy'] for k, _, _ in ORDER]
    tsd   = [df.loc[k, 'tardy_sd'] for k, _, _ in ORDER]
    cols  = [a[2] for a in ORDER]
    x = np.arange(len(labels))

    ax_c.bar(x, comps, 0.7, yerr=csd, color=cols,
             edgecolor='black', linewidth=1.0, capsize=4,
             error_kw={'elinewidth': 1.3, 'capthick': 1.3})
    pad = (max(comps) - min(comps) + 4) * 0.05
    for i, (m, s) in enumerate(zip(comps, csd)):
        ax_c.text(i, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                  fontsize=11, fontweight='bold')
    ax_c.set_xticks(x); ax_c.set_xticklabels(labels, rotation=20, ha='right', fontsize=11)
    ax_c.set_ylabel('Completion rate (%)', fontsize=13)
    ax_c.set_title('(a) Completion across architecture variants',
                   fontsize=14, fontweight='bold')
    ax_c.set_ylim(85, 102); ax_c.grid(alpha=0.3, axis='y'); ax_c.tick_params(axis='y', labelsize=11)

    ax_t.bar(x, tars, 0.7, yerr=tsd, color=cols,
             edgecolor='black', linewidth=1.0, capsize=4,
             error_kw={'elinewidth': 1.3, 'capthick': 1.3})
    pad = (max(tars) + 1) * 0.05
    for i, (m, s) in enumerate(zip(tars, tsd)):
        ax_t.text(i, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                  fontsize=11, fontweight='bold')
    ax_t.set_xticks(x); ax_t.set_xticklabels(labels, rotation=20, ha='right', fontsize=11)
    ax_t.set_ylabel('Tardiness rate (%)', fontsize=13)
    ax_t.set_title('(b) Tardiness across architecture variants',
                   fontsize=14, fontweight='bold')
    ax_t.set_ylim(0, max(tars) * 1.4 + 1); ax_t.grid(alpha=0.3, axis='y'); ax_t.tick_params(axis='y', labelsize=11)

    fig.tight_layout()
    out_path = out('Figure_09_Architecture_Variants.png')
    fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
