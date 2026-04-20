"""Figure 6 -- 4-panel ablation suite.

  python reproduce/reproduce_fig6_ablation.py \
         --data data/industrial_servers_2024.csv --seeds 10

(a) Component ablation @ 4-bank/10-day
(b) Component ablation @ 1-bank/10-day
(c) TLO mechanism ablation
(d) Reward-scale insensitivity (line chart, 100x range)

The full sweep retrains and evaluates every variant; aggregated
results are written to results/ablation_*.csv. Pass `--use-cache`
to plot directly from a prior run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if THIS not in sys.path:
    sys.path.insert(0, THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _common import (RESULTS, COLOR_LHAC, COLOR_GRAY_LIGHT,
                     COLOR_GRAY_FAINT, COLOR_GRAY_DARK,
                     LINE_PALETTE, out)
from _pipeline import ablation_sweep

VARIANT_ORDER = ['full_daf', 'case_fpr', 'dap_case', 'dap_fpr',
                 'case_only', 'dap_only', 'fpr_only', 'baseline_lhac']
VARIANT_LABEL = {
    'full_daf':      'All',
    'case_fpr':      'No DAP',
    'dap_case':      'No FPR',
    'dap_fpr':       'No CASE',
    'case_only':     'CASE only',
    'dap_only':      'DAP only',
    'fpr_only':      'FPR only',
    'baseline_lhac': 'Baseline',
}

TLO_ORDER = ['adaptive_tlo', 'fixed_tlo_05', 'fixed_tlo_10',
             'fixed_tlo_20', 'no_tlo', 'tlo_in_critic']
TLO_LABEL = {
    'adaptive_tlo':  'Adaptive\n(LHAC)',
    'fixed_tlo_05':  r'Fixed $\tau$=0.05',
    'fixed_tlo_10':  r'Fixed $\tau$=0.10',
    'fixed_tlo_20':  r'Fixed $\tau$=0.20',
    'no_tlo':        r'No TLO ($\tau$$\to\infty$)',
    'tlo_in_critic': 'TLO in critic',
}


def panel_component(ax, sub, title, ylim):
    sub = sub.set_index('variant')
    means = [sub.loc[v, 'completion'] for v in VARIANT_ORDER]
    sds   = [sub.loc[v, 'completion_sd'] for v in VARIANT_ORDER]
    labels = [VARIANT_LABEL[v] for v in VARIANT_ORDER]
    colors = [COLOR_LHAC] + [COLOR_GRAY_LIGHT] * 6 + [COLOR_GRAY_FAINT]
    x = np.arange(len(labels))
    ax.bar(x, means, 0.7, yerr=sds, color=colors,
           edgecolor='black', linewidth=1.0, capsize=4,
           error_kw={'elinewidth': 1.2, 'capthick': 1.2})
    pad = (ylim[1] - ylim[0]) * 0.04
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + pad, f'{m:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Completion (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(*ylim); ax.grid(alpha=0.3, axis='y'); ax.tick_params(axis='y', labelsize=10)


def panel_tlo(ax, df_tlo):
    df_tlo = df_tlo.set_index('variant')
    labels = [TLO_LABEL[k] for k in TLO_ORDER]
    comps = [df_tlo.loc[k, 'completion'] for k in TLO_ORDER]
    csd   = [df_tlo.loc[k, 'completion_sd'] for k in TLO_ORDER]
    tars  = [df_tlo.loc[k, 'tardy'] for k in TLO_ORDER]
    tsd   = [df_tlo.loc[k, 'tardy_sd'] for k in TLO_ORDER]
    cc = [COLOR_LHAC] + [COLOR_GRAY_LIGHT] * (len(TLO_ORDER) - 1)
    ct = [COLOR_LHAC] + [COLOR_GRAY_DARK]  * (len(TLO_ORDER) - 1)
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, comps, w, yerr=csd, color=cc,
           edgecolor='black', linewidth=1.0, capsize=4,
           error_kw={'elinewidth': 1.2, 'capthick': 1.2}, label='Completion')
    ax2 = ax.twinx()
    ax2.bar(x + w/2, tars, w, yerr=tsd, color=ct,
            edgecolor='black', linewidth=1.0, capsize=4,
            alpha=0.55,
            error_kw={'elinewidth': 1.2, 'capthick': 1.2}, label='Tardiness')
    cy = (90, 103); ty = (0, 9)
    ax.set_ylim(*cy); ax2.set_ylim(*ty)
    p_c = (cy[1] - cy[0]) * 0.04; p_t = (ty[1] - ty[0]) * 0.04
    for i, (c, sc) in enumerate(zip(comps, csd)):
        ax.text(i - w/2, c + sc + p_c, f'{c:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=COLOR_LHAC if i == 0 else 'black')
    for i, (t, st) in enumerate(zip(tars, tsd)):
        ax2.text(i + w/2, t + st + p_t, f'{t:.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#444')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Completion (%)', fontsize=12)
    ax2.set_ylabel('Tardiness (%)', fontsize=12, rotation=270, labelpad=15)
    ax.set_title('(c) TLO mechanism ablation', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10); ax2.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3, axis='y')


def panel_reward_scale(ax, df_rs):
    methods = ['adaptive_tlo', 'lppo', 'envelope_morl',
               'ppo_lagrangian', 'rcpo', 'tlo_in_critic']
    method_label = {
        'adaptive_tlo':   'LHAC (adaptive TLO)',
        'lppo':           'LPPO (fixed TLO)',
        'envelope_morl':  'Envelope MORL',
        'ppo_lagrangian': 'PPO-Lagrangian',
        'rcpo':           'RCPO',
        'tlo_in_critic':  'TLO in critic',
    }
    markers = {'adaptive_tlo': 'o', 'lppo': 's', 'envelope_morl': '^',
               'ppo_lagrangian': 'D', 'rcpo': 'v', 'tlo_in_critic': 'P'}
    for k in methods:
        sub = df_rs[df_rs['method'] == k].sort_values('scale')
        ax.plot(sub['scale'], sub['completion'], marker=markers[k],
                color=LINE_PALETTE[k], lw=2.4, ms=8, label=method_label[k])
    ax.set_xscale('log')
    ax.set_xlabel(r'Reward-scale factor (x nominal)', fontsize=12)
    ax.set_ylabel('Completion (%)', fontsize=12)
    ax.set_title('(d) Reward-scale insensitivity (100x range)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(91, 100); ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=10)
    ax.legend(loc='lower center', fontsize=9, ncol=2, frameon=True, framealpha=0.95)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/industrial_servers_2024.csv')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--use-cache', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = out('Figure_06_Ablation.png')
    p_c = os.path.join(RESULTS, 'ablation_components.csv')
    p_t = os.path.join(RESULTS, 'ablation_tlo.csv')
    p_r = os.path.join(RESULTS, 'ablation_reward_scale.csv')

    if not args.use_cache:
        if os.path.exists(args.data):
            print(f'[load] using confidential dataset: {args.data}')
        else:
            print(f'[load] {args.data} not present; using calibrated synthetic'
                  f' instances')
        print(f'[sweep] component / TLO / reward-scale ablations x {args.seeds} seeds')
        t0 = time.time()
        ablation_sweep(seeds=args.seeds,
                       out_components=p_c, out_tlo=p_t, out_rs=p_r)
        print(f'\n[done] ablation sweep completed in {(time.time() - t0)/60:.1f} min')
    else:
        print('[cache] reading aggregated ablation statistics...')

    df_comp = pd.read_csv(p_c)
    df_tlo  = pd.read_csv(p_t)
    df_rs   = pd.read_csv(p_r)

    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('white')
    panel_component(a,
                    df_comp[(df_comp['banks'] == 4) & (df_comp['window'] == '10d')],
                    '(a) Component ablation -- 4-bank / 10-day', (88, 102))
    panel_component(b,
                    df_comp[(df_comp['banks'] == 1) & (df_comp['window'] == '10d')],
                    '(b) Component ablation -- 1-bank / 10-day', (30, 56))
    panel_tlo(c, df_tlo)
    panel_reward_scale(d, df_rs)
    fig.tight_layout(pad=2.5)
    fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.25,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
