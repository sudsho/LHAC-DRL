"""Weighted-reward single-agent baseline.

Trains a standard PPO actor-critic on a scalar combination of the
two reward streams: r_eff = w * r_completion + (1 - w) * r_tardiness.
Used as the simplest possible MORL baseline (Roijers et al., 2013).
The weight `w` is exposed as a hyperparameter so it can be swept in
the reward-scale insensitivity experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from lhac.env import LHACEnv
from .morl_common import _BaseScheduler, collect_rollout, gae


@dataclass
class WeightedConfig:
    actor_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.20
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    completion_weight: float = 0.85    # convex combination weight w


class WeightedRLScheduler(_BaseScheduler):
    """Single-agent PPO on the scalarised reward."""

    def __init__(self, n_actions: int, config: Optional[WeightedConfig] = None,
                 device: Optional[str] = None):
        super().__init__(n_actions, device)
        self.cfg = config or WeightedConfig()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.actor_lr)

    def train(self, env_factory: Callable[[int], LHACEnv], episodes: int):
        w = self.cfg.completion_weight
        for ep in range(episodes):
            env = env_factory(ep)
            buf = collect_rollout(self.policy, env, self.device)
            shaped = [w * rc + (1.0 - w) * rt
                      for rc, rt in zip(buf.r_completion, buf.r_tardiness)]
            adv, ret = gae(shaped, buf.values_completion, buf.dones,
                           self.cfg.gamma, self.cfg.lam)
            states = torch.tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
            cells  = torch.tensor(np.array(buf.cell_feats), dtype=torch.float32, device=self.device)
            masks  = torch.tensor(np.array(buf.masks), dtype=torch.float32, device=self.device)
            acts   = torch.tensor(buf.actions, dtype=torch.long, device=self.device)
            old_lp = torch.tensor(buf.logps, dtype=torch.float32, device=self.device)
            adv_t  = torch.tensor(adv, dtype=torch.float32, device=self.device)
            ret_t  = torch.tensor(ret, dtype=torch.float32, device=self.device)

            for _ in range(self.cfg.epochs):
                idxs = np.random.permutation(len(buf.actions))
                for start in range(0, len(idxs), self.cfg.batch_size):
                    mb = torch.tensor(idxs[start:start + self.cfg.batch_size],
                                      dtype=torch.long, device=self.device)
                    logits, v = self.policy(states[mb], cells[mb], masks[mb])
                    dist = Categorical(logits=logits)
                    lp = dist.log_prob(acts[mb])
                    ratio = (lp - old_lp[mb]).exp()
                    clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                    pol_loss = -torch.min(ratio * adv_t[mb], clipped * adv_t[mb]).mean()
                    val_loss = F.mse_loss(v, ret_t[mb])
                    entropy = dist.entropy().mean()
                    loss = pol_loss + 0.5 * val_loss - self.cfg.entropy_coef * entropy
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.opt.step()
