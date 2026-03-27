"""DQN+DAF baseline.

A standard Double-DQN with the same Deferred Action Framework
augmentations (DAP, CASE, FPR) but a flat single-objective scalar
reward r1 + lambda * r2 (no lexicographic ordering). Used in the
architecture-variant comparison of Section 5.5 / Figure 9.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lhac.env import LHACEnv
from lhac.networks import CASEEncoder


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 5_000
    buffer_size: int = 50_000
    batch_size: int = 64
    target_sync: int = 200
    lambda_tardy: float = 0.30


class _QNet(nn.Module):
    def __init__(self, state_dim, cell_dim, n_actions, hidden=256, d_model=128):
        super().__init__()
        self.encoder = CASEEncoder(state_dim, cell_dim, d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s, c, m):
        h = self.encoder(s, c)
        q = self.head(h)
        return q + (m + 1e-9).log()


class DQNDAFScheduler:
    def __init__(
        self,
        n_actions: int,
        state_dim: int = 78,
        cell_dim: int = 12,
        config: Optional[DQNConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or DQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions
        self.q = _QNet(state_dim, cell_dim, n_actions).to(self.device)
        self.q_target = _QNet(state_dim, cell_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = deque(maxlen=self.cfg.buffer_size)
        self.steps = 0

    # ----- inference ---------------------------------------------------------

    @torch.no_grad()
    def act(self, env: LHACEnv, eps: float = 0.0) -> int:
        feas = env.feasibility_mask()
        if random.random() < eps:
            choices = [i for i, v in enumerate(feas) if v > 0]
            return int(random.choice(choices))
        s = torch.tensor(env.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        c = torch.tensor(env.per_cell_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.tensor(feas, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s, c, m)
        return int(q.argmax(-1).item())

    def schedule(self, env: LHACEnv) -> dict:
        env.reset()
        done = False
        while not done:
            a = self.act(env, eps=0.0)
            _s, _m, _r1, _r2, done, _i = env.step(a)
        return env.summary()
