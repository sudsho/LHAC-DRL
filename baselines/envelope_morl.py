"""Envelope Multi-Objective Reinforcement Learning.

Yang, Sun, and Narasimhan (2019). Envelope MORL maintains a
vector-valued Q-function `Q(s, a, w)` that is jointly optimised
across a distribution of preference weights `w`. At inference time
the user supplies a single preference (here, the lexicographic-
completion-then-tardiness preference) and the greedy action is
`argmax_a w^T Q(s, a, w)`.

The implementation below uses a 2-objective formulation
(completion, negative-tardiness) and a CASE-encoded state
representation to remain comparable with the LHAC architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lhac.env import LHACEnv
from lhac.networks import CASEEncoder
from .morl_common import _BaseScheduler


# ---------------------------------------------------------------------------
# Vector-valued Q-network
# ---------------------------------------------------------------------------

class _EnvelopeQ(nn.Module):
    """Q-network with N output heads, one per objective.

    Inputs are state, per-cell features, action mask, and a length-N
    preference vector. The preference is concatenated with the
    encoded state so the same network parameterises Q for any weight.
    """

    def __init__(self, state_dim=78, cell_feat_dim=12, n_actions=55,
                 n_objectives=2, d_model=128, hidden=256):
        super().__init__()
        self.n_objectives = n_objectives
        self.encoder = CASEEncoder(state_dim, cell_feat_dim, d_model=d_model)
        self.preference_proj = nn.Linear(n_objectives, d_model)
        self.q_head = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions * n_objectives),
        )
        self.n_actions = n_actions

    def forward(self, state, cell_feats, mask, weight) -> torch.Tensor:
        h = self.encoder(state, cell_feats)
        w = self.preference_proj(weight)
        joint = torch.cat([h, w], dim=-1)
        q = self.q_head(joint)
        q = q.view(-1, self.n_actions, self.n_objectives)
        # Mask infeasible actions on the scalar projection
        q = q + (mask + 1e-9).log().unsqueeze(-1)
        return q


@dataclass
class EnvelopeConfig:
    lr: float = 5e-4
    gamma: float = 0.99
    target_sync: int = 200
    homotopy_steps: int = 1000
    n_objectives: int = 2


class EnvelopeMORLScheduler(_BaseScheduler):
    """Envelope MORL with linear-preference inference.

    The preference vector during evaluation is `[1.0, 0.0]` for the
    lexicographic completion-first variant and `[0.5, 0.5]` for the
    balanced reference. The training loop performs homotopy over
    sampled weights to encourage envelope coverage of the Pareto
    front (Section 3 of Yang et al. 2019).
    """

    def __init__(self, n_actions: int, config: Optional[EnvelopeConfig] = None,
                 device: Optional[str] = None):
        super().__init__(n_actions, device)
        self.cfg = config or EnvelopeConfig()
        self.q = _EnvelopeQ(n_actions=n_actions,
                            n_objectives=self.cfg.n_objectives).to(self.device)
        self.q_target = _EnvelopeQ(n_actions=n_actions,
                                   n_objectives=self.cfg.n_objectives).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self._step = 0

    def _sample_weight(self) -> torch.Tensor:
        """Dirichlet-distributed preference weights for homotopy."""
        w = np.random.dirichlet(np.ones(self.cfg.n_objectives))
        return torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _bellman_target(self, q_next: torch.Tensor, reward_vec: torch.Tensor,
                        done: bool) -> torch.Tensor:
        gamma = 0 if done else self.cfg.gamma
        # Vector-valued Bellman update
        max_q_per_obj, _ = q_next.max(dim=1)
        target = reward_vec + gamma * max_q_per_obj
        return target.detach()

    def train(self, env_factory: Callable[[int], LHACEnv], episodes: int):
        for ep in range(episodes):
            env = env_factory(ep)
            state, mask = env.reset()
            cell = env.per_cell_features()
            done = False
            weight = self._sample_weight()
            while not done:
                s = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                c = torch.tensor(cell, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                m = torch.tensor(mask, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)

                q = self.q(s, c, m, weight)
                scalar_q = (q * weight.unsqueeze(1)).sum(dim=-1)
                if np.random.rand() < 0.05:
                    feas = [i for i, v in enumerate(mask) if v > 0]
                    a = int(np.random.choice(feas))
                else:
                    a = int(scalar_q.argmax(-1).item())

                next_state, next_mask, r1, r2, done, _ = env.step(a)
                next_cell = env.per_cell_features()
                reward_vec = torch.tensor([r1, r2], dtype=torch.float32,
                                          device=self.device)
                with torch.no_grad():
                    n_s = torch.tensor(next_state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    n_c = torch.tensor(next_cell, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    n_m = torch.tensor(next_mask, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    q_next = self.q_target(n_s, n_c, n_m, weight).squeeze(0)
                target_vec = self._bellman_target(q_next, reward_vec, done)

                # Vector-valued temporal-difference loss
                pred = q.squeeze(0)[a]
                td_err = pred - target_vec
                loss = td_err.pow(2).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                self.opt.step()

                state, mask, cell = next_state, next_mask, next_cell
                self._step += 1
                if self._step % self.cfg.target_sync == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def schedule(self, env: LHACEnv) -> dict:
        # Lexicographic preference: completion first.
        weight = torch.tensor([1.0, 0.0], dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        env.reset()
        done = False
        while not done:
            s = torch.tensor(env.observe(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            c = torch.tensor(env.per_cell_features(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            m = torch.tensor(env.feasibility_mask(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            q = self.q(s, c, m, weight)
            scalar_q = (q * weight.unsqueeze(1)).sum(dim=-1)
            a = int(scalar_q.argmax(-1).item())
            _s, _m, _r1, _r2, done, _i = env.step(a)
        return env.summary()
