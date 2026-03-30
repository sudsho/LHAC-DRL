"""Lexicographic PPO (LPPO) with a fixed threshold.

Zhang, Lin, Han, and Lv (2023). LPPO uses two actor-critic agents
arranged in a strict lexicographic order, just like LHAC, but with
a hand-tuned constant threshold tau (no adaptation). The candidate
set produced by the primary policy is

    A_hat = { a : pi_1(a|s) >= pi_1(argmax|s) - tau }

i.e. an additive cut-off, not the relative cut-off of LHAC. The
secondary policy then samples within A_hat to optimise tardiness.

The implementation reuses the LHAC PPO trainer for the primary loop
and substitutes the additive-threshold filter at action selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from lhac.env import LHACEnv
from lhac.networks import ActorCritic
from .morl_common import _BaseScheduler


@dataclass
class LPPOConfig:
    actor_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.20
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    tau: float = 0.10  # additive threshold; fixed throughout training


class LPPOScheduler(_BaseScheduler):
    """Lexicographic PPO with a fixed-threshold filter.

    Two actor-critic agents share the same trunk but have separate
    heads. The primary maximises completion; the secondary chooses
    among the primary's near-optimal actions to minimise tardiness.
    """

    def __init__(self, n_actions: int, config: Optional[LPPOConfig] = None,
                 device: Optional[str] = None):
        super().__init__(n_actions, device)
        self.cfg = config or LPPOConfig()
        self.policy_secondary = ActorCritic(n_actions=n_actions).to(self.device)
        self.opt_primary = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.actor_lr)
        self.opt_secondary = torch.optim.Adam(self.policy_secondary.parameters(),
                                              lr=self.cfg.actor_lr)

    def _additive_filter(self, primary_probs: torch.Tensor,
                         feasible: torch.Tensor) -> torch.Tensor:
        max_p = primary_probs.max()
        cutoff = max_p - self.cfg.tau
        keep = (primary_probs >= cutoff).float() * feasible
        if keep.sum() < 1.0:
            keep = feasible.clone().float()
        return keep

    @torch.no_grad()
    def schedule(self, env: LHACEnv) -> dict:
        env.reset()
        done = False
        while not done:
            s = torch.tensor(env.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
            c = torch.tensor(env.per_cell_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.tensor(env.feasibility_mask(), dtype=torch.float32, device=self.device).unsqueeze(0)
            l1, _ = self.policy(s, c, m)
            l2, _ = self.policy_secondary(s, c, m)
            p1 = F.softmax(l1, dim=-1).squeeze(0)
            p2 = F.softmax(l2, dim=-1).squeeze(0)
            cand = self._additive_filter(p1 * m.squeeze(0), m.squeeze(0))
            scored = (p2 * cand)
            scored = scored / max(float(scored.sum()), 1e-8)
            a = int(scored.argmax().item())
            _s, _m, _r1, _r2, done, _i = env.step(a)
        return env.summary()
