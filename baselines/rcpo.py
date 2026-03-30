"""Reward Constrained Policy Optimisation (RCPO).

Tessler, Mankowitz, and Mannor (2019). RCPO modifies the reward
signal seen by the policy with a penalty term proportional to the
constraint violation, and uses a slowly-updated multiplier to
balance the two terms. Unlike PPO-Lagrangian, the penalty enters
the reward at every step rather than as a separate dual update on
the entire trajectory.
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
class RCPOConfig:
    actor_lr: float = 3e-4
    multiplier_lr: float = 1e-3
    gamma: float = 0.99
    lam_gae: float = 0.95
    clip_eps: float = 0.20
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    cost_limit: float = 0.05


class RCPOScheduler(_BaseScheduler):
    """Reward-Constrained Policy Optimisation scheduler.

    The shaped per-step reward is `r_eff = r_completion - mu * c`,
    where `c` is the realised tardiness signal and `mu` is the
    Lagrange multiplier updated as a function of running cost
    violations. RCPO is sample-efficient because the penalty enters
    the value function and propagates through bootstrapped returns.
    """

    def __init__(self, n_actions: int, config: Optional[RCPOConfig] = None,
                 device: Optional[str] = None):
        super().__init__(n_actions, device)
        self.cfg = config or RCPOConfig()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.actor_lr)
        self.mu = 0.5
        self._cost_avg = 0.0

    def _update_multiplier(self, batch_cost: float) -> None:
        # Exponential moving average of cost violations, then a
        # gradient-style update on the multiplier.
        beta = 0.10
        self._cost_avg = (1.0 - beta) * self._cost_avg + beta * batch_cost
        violation = self._cost_avg - self.cfg.cost_limit
        self.mu = float(max(0.0, self.mu + self.cfg.multiplier_lr * violation))

    def train(self, env_factory: Callable[[int], LHACEnv], episodes: int):
        for ep in range(episodes):
            env = env_factory(ep)
            buf = collect_rollout(self.policy, env, self.device)

            # Shape per-step rewards with the current multiplier
            shaped = [
                rc - self.mu * (-rt)
                for rc, rt in zip(buf.r_completion, buf.r_tardiness)
            ]
            adv, ret = gae(shaped, buf.values_completion, buf.dones,
                           self.cfg.gamma, self.cfg.lam_gae)

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
                    policy_loss = -torch.min(ratio * adv_t[mb], clipped * adv_t[mb]).mean()
                    value_loss = F.mse_loss(v, ret_t[mb])
                    entropy = dist.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.opt.step()

            self._update_multiplier(float(np.mean([-r for r in buf.r_tardiness])))
