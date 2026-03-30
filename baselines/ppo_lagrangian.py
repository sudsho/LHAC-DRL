"""PPO-Lagrangian for constrained scheduling.

Stooke, Achiam, and Abbeel (2020). The completion-rate objective is
optimised with the standard clipped PPO surrogate; the tardiness
constraint is enforced through a Lagrangian dual variable that is
updated by a slow gradient ascent on the constraint violation.

The dual update follows the "responsive safety" parameterisation
where lambda is parameterised through a softplus of an unconstrained
real-valued variable so that lambda >= 0 is preserved automatically.
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
class PPOLagrangianConfig:
    actor_lr: float = 3e-4
    lambda_lr: float = 5e-3
    gamma: float = 0.99
    lam_gae: float = 0.95
    clip_eps: float = 0.20
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    cost_limit: float = 0.05    # tardiness budget (fraction of servers)


class PPOLagrangianScheduler(_BaseScheduler):
    """PPO-Lagrangian scheduler.

    The Lagrangian is updated on the cost-budget violation
    L(theta, lambda) = E[R_completion(s, a)] - lambda * (E[C_tardy] - d),
    where d = `cost_limit` is the per-episode tardiness budget."""

    def __init__(self, n_actions: int,
                 config: Optional[PPOLagrangianConfig] = None,
                 device: Optional[str] = None):
        super().__init__(n_actions, device)
        self.cfg = config or PPOLagrangianConfig()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.actor_lr)
        # Unconstrained dual parameter; effective lambda = softplus(rho)
        self._rho = torch.nn.Parameter(torch.tensor(np.log(np.expm1(1.0)),
                                                    dtype=torch.float32, device=self.device))
        self.lambda_opt = torch.optim.Adam([self._rho], lr=self.cfg.lambda_lr)

    @property
    def lam(self) -> float:
        return float(F.softplus(self._rho).item())

    def train(self, env_factory: Callable[[int], LHACEnv], episodes: int):
        """Joint actor and dual training loop."""
        for ep in range(episodes):
            env = env_factory(ep)
            buf = collect_rollout(self.policy, env, self.device)
            adv, ret = gae(buf.r_completion, buf.values_completion, buf.dones,
                           self.cfg.gamma, self.cfg.lam_gae)
            cost_signal = float(np.mean(buf.r_tardiness))
            constraint_violation = (-cost_signal) - self.cfg.cost_limit

            states = torch.tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
            cells  = torch.tensor(np.array(buf.cell_feats), dtype=torch.float32, device=self.device)
            masks  = torch.tensor(np.array(buf.masks), dtype=torch.float32, device=self.device)
            acts   = torch.tensor(buf.actions, dtype=torch.long, device=self.device)
            old_lp = torch.tensor(buf.logps, dtype=torch.float32, device=self.device)
            adv_t  = torch.tensor(adv, dtype=torch.float32, device=self.device)
            ret_t  = torch.tensor(ret, dtype=torch.float32, device=self.device)

            # Penalised advantage: A_eff = A_R - lambda * A_C
            adv_eff = adv_t - self.lam * (-torch.tensor(buf.r_tardiness,
                                                        dtype=torch.float32, device=self.device))

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
                    policy_loss = -torch.min(ratio * adv_eff[mb], clipped * adv_eff[mb]).mean()
                    value_loss = F.mse_loss(v, ret_t[mb])
                    entropy = dist.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.opt.step()

            # Dual ascent on constraint violation
            rho_loss = -F.softplus(self._rho) * float(constraint_violation)
            self.lambda_opt.zero_grad()
            rho_loss.backward()
            self.lambda_opt.step()
