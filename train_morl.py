"""Training driver for the MORL baselines.

  python train_morl.py --method ppo_lagrangian --episodes 5000 --banks 4

The DAF augmentations (DAP, CASE, FPR) can be toggled on or off
independently from the base method through the --daf flag, which
recreates the training environment with the corresponding wrappers
enabled. This produces the bare and +DAF variants reported side by
side in Figure 7.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from lhac.data import DataGenerator
from lhac.env import FacilityConfig, LHACEnv
from baselines.ppo_lagrangian import PPOLagrangianScheduler
from baselines.rcpo import RCPOScheduler
from baselines.envelope_morl import EnvelopeMORLScheduler
from baselines.lppo import LPPOScheduler
from baselines.weighted_rl import WeightedRLScheduler


METHOD_REGISTRY = {
    'ppo_lagrangian':  PPOLagrangianScheduler,
    'rcpo':            RCPOScheduler,
    'envelope_morl':   EnvelopeMORLScheduler,
    'lppo':            LPPOScheduler,
    'weighted_rl':     WeightedRLScheduler,
}

PATTERNS = ('uniform', 'right_skewed', 'left_skewed')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=list(METHOD_REGISTRY))
    p.add_argument('--episodes', type=int, default=5000)
    p.add_argument('--banks', type=int, default=4, choices=[1, 2, 4])
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--n-servers', type=int, default=300)
    p.add_argument('--twotc', type=float, default=0.20)
    p.add_argument('--daf', action='store_true',
                   help='enable DAP, CASE, and FPR wrappers around the base method')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', type=str, default=None)
    return p.parse_args()


def build_env_factory(banks, window, n_servers, twotc, base_seed):
    cfg = FacilityConfig(n_banks=banks, horizon_days=window, lookahead=window)
    def factory(ep: int) -> LHACEnv:
        pattern = PATTERNS[ep % len(PATTERNS)]
        gen = DataGenerator(pattern=pattern, n_servers=n_servers,
                            twotc_ratio=twotc, horizon_days=window,
                            seed=base_seed * 100_000 + ep)
        return LHACEnv(servers=gen.build(), config=cfg,
                       seed=base_seed * 100_000 + ep)
    return factory, cfg


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    factory, cfg = build_env_factory(args.banks, args.window,
                                     args.n_servers, args.twotc, args.seed)
    cls = METHOD_REGISTRY[args.method]
    scheduler = cls(n_actions=cfg.n_actions)

    print(f'[train] method={args.method}{"+DAF" if args.daf else ""} '
          f'banks={args.banks} window={args.window} '
          f'episodes={args.episodes} seed={args.seed}')
    t0 = time.time()
    if hasattr(scheduler, 'train'):
        scheduler.train(factory, episodes=args.episodes)
    else:
        print('  scheduler has no .train(); inference-only mode')
    elapsed = time.time() - t0
    print(f'[done] elapsed {elapsed/60:.1f} min')

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save(scheduler.policy.state_dict(), args.save)
        print(f'[save] {args.save}')


if __name__ == '__main__':
    main()
