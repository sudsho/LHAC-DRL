"""Shared scaffolding for the multi-objective RL baselines.

Each MORL method in this package follows the same outer protocol so
that the reproduction pipeline can swap them through a common
factory: every class exposes `train(env_factory, episodes)` and
`schedule(env)`. The training loops are method-specific (Lagrangian
duals, scalarisation weights, vector-valued Q-functions, etc.) but
share the rollout buffer and PPO clipped surrogate of Schulman et al.
(2017) when applicable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from lhac.env import LHACEnv
from lhac.networks import ActorCritic, CASEEncoder


@dataclass
class MORLRollout:
    """Storage for a single multi-objective trajectory."""
    states: List[np.ndarray] = field(default_factory=list)
    cell_feats: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    r_completion: List[float] = field(default_factory=list)
    r_tardiness:  List[float] = field(default_factory=list)
    values_completion: List[float] = field(default_factory=list)
    values_tardiness:  List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


def collect_rollout(policy: ActorCritic, env: LHACEnv,
                    device: torch.device) -> MORLRollout:
    """Drive a single episode under the given policy.

    The collected trajectory exposes both reward channels separately
    (completion and tardiness) so each MORL method can apply its own
    scalarisation / constraint formulation downstream.
    """
    state, mask = env.reset()
    cell = env.per_cell_features()
    buf = MORLRollout()
    done = False
    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(cell, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, v = policy(s, c, m)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            dist = Categorical(probs=probs)
            a = int(dist.sample().item())
            lp = float(torch.log(probs[a] + 1e-12).item())
        next_state, next_mask, r1, r2, done, _ = env.step(a)
        buf.states.append(state); buf.cell_feats.append(cell)
        buf.masks.append(mask); buf.actions.append(a); buf.logps.append(lp)
        buf.r_completion.append(float(r1))
        buf.r_tardiness.append(float(r2))
        buf.values_completion.append(float(v.item()))
        buf.values_tardiness.append(0.0)
        buf.dones.append(bool(done))
        state, mask, cell = next_state, next_mask, env.per_cell_features()
    return buf


def gae(rewards: List[float], values: List[float], dones: List[bool],
        gamma: float = 0.99, lam: float = 0.95):
    """Standard generalised advantage estimation."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
        next_v = values[t]
    ret = adv + np.asarray(values, dtype=np.float32)
    if adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


class _BaseScheduler:
    """Inference shell shared by every MORL scheduler.

    Subclasses customise `train()` but use this class for the
    deterministic action-selection path that the reproduction
    pipeline drives during evaluation.
    """

    def __init__(self, n_actions: int, device: Optional[str] = None):
        self.n_actions = n_actions
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.policy = ActorCritic(n_actions=n_actions).to(self.device)

    @torch.no_grad()
    def schedule(self, env: LHACEnv) -> dict:
        env.reset()
        done = False
        while not done:
            s = torch.tensor(env.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
            c = torch.tensor(env.per_cell_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.tensor(env.feasibility_mask(), dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.policy(s, c, m)
            a = int(logits.argmax(-1).item())
            _s, _m, _r1, _r2, done, _i = env.step(a)
        return env.summary()
