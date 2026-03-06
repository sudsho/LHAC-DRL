"""PPO trainer with dual-agent updates for the LHAC architecture.

Both agents share the same trajectory data but receive separate
advantage signals: Agent 1 uses the FPR-shaped completion reward
r_tilde_1 (Eq. 9); Agent 2 uses the proportional tardiness reward
r_2 (Eq. 4). The actor-critic update follows the standard clipped
PPO surrogate (Schulman et al., 2017; Eq. 10 of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import LHACEnv
from .fpr import feasibility_potential, shaped_reward
from .networks import ActorCritic
from .tlo import AdaptiveTLOFilter


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.20
    epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    beta_tau: float = 0.10


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    states: List[np.ndarray] = field(default_factory=list)
    cell_feats: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps1: List[float] = field(default_factory=list)
    logps2: List[float] = field(default_factory=list)
    r1: List[float] = field(default_factory=list)
    r2: List[float] = field(default_factory=list)
    v1: List[float] = field(default_factory=list)
    v2: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Joint training loop for the two LHAC actor-critics."""

    def __init__(
        self,
        agent1: ActorCritic,
        agent2: ActorCritic,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or PPOConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent1 = agent1.to(self.device)
        self.agent2 = agent2.to(self.device)
        self.opt1 = torch.optim.Adam(self.agent1.parameters(), lr=self.cfg.actor_lr)
        self.opt2 = torch.optim.Adam(self.agent2.parameters(), lr=self.cfg.actor_lr)
        self.tlo = AdaptiveTLOFilter(beta=self.cfg.beta_tau)

    # ----- rollout collection -----------------------------------------------

    def rollout(self, env: LHACEnv) -> Rollout:
        buf = Rollout()
        state, mask = env.reset()
        cell = env.per_cell_features()
        phi = feasibility_potential(env)
        self.tlo.reset()
        done = False

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_t = torch.tensor(cell, dtype=torch.float32, device=self.device).unsqueeze(0)
            m_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits1, v1 = self.agent1(s_t, c_t, m_t)
                logits2, v2 = self.agent2(s_t, c_t, m_t)
                p1 = F.softmax(logits1, dim=-1).squeeze(0)
                p2 = F.softmax(logits2, dim=-1).squeeze(0)
                feasible = m_t.squeeze(0)
                a, _tau = self.tlo.select(p1, p2, feasible, deterministic=False)
                lp1 = float(torch.log(p1[a] + 1e-12).item())
                lp2 = float(torch.log(p2[a] + 1e-12).item())

            next_state, next_mask, r1, r2, done, _info = env.step(a)
            next_cell = env.per_cell_features()
            phi_next = feasibility_potential(env)
            r_tilde = shaped_reward(r1, phi, phi_next, self.cfg.gamma)

            buf.states.append(state)
            buf.cell_feats.append(cell)
            buf.masks.append(mask)
            buf.actions.append(int(a))
            buf.logps1.append(lp1)
            buf.logps2.append(lp2)
            buf.r1.append(r_tilde)
            buf.r2.append(float(r2))
            buf.v1.append(float(v1.item()))
            buf.v2.append(float(v2.item()))
            buf.dones.append(bool(done))

            state, mask, cell, phi = next_state, next_mask, next_cell, phi_next
        return buf

    # ----- GAE -------------------------------------------------------------------

    def _gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        next_v = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_v * mask - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * mask * last_gae
            adv[t] = last_gae
            next_v = values[t]
        ret = adv + np.array(values, dtype=np.float32)
        # standardize advantage
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # ----- update -------------------------------------------------------------

    def update(self, buf: Rollout) -> dict:
        T = len(buf.actions)
        if T == 0:
            return {"n": 0}

        adv1, ret1 = self._gae(buf.r1, buf.v1, buf.dones)
        adv2, ret2 = self._gae(buf.r2, buf.v2, buf.dones)

        states = torch.tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
        cells  = torch.tensor(np.array(buf.cell_feats), dtype=torch.float32, device=self.device)
        masks  = torch.tensor(np.array(buf.masks), dtype=torch.float32, device=self.device)
        acts   = torch.tensor(buf.actions, dtype=torch.long, device=self.device)
        old_lp1 = torch.tensor(buf.logps1, dtype=torch.float32, device=self.device)
        old_lp2 = torch.tensor(buf.logps2, dtype=torch.float32, device=self.device)
        adv1_t = torch.tensor(adv1, dtype=torch.float32, device=self.device)
        adv2_t = torch.tensor(adv2, dtype=torch.float32, device=self.device)
        ret1_t = torch.tensor(ret1, dtype=torch.float32, device=self.device)
        ret2_t = torch.tensor(ret2, dtype=torch.float32, device=self.device)

        loss1 = self._ppo_step(self.agent1, self.opt1, states, cells, masks,
                               acts, old_lp1, adv1_t, ret1_t)
        loss2 = self._ppo_step(self.agent2, self.opt2, states, cells, masks,
                               acts, old_lp2, adv2_t, ret2_t)
        return {"loss_agent1": loss1, "loss_agent2": loss2, "n": T, "tau": self.tlo.tau}

    def _ppo_step(self, agent, opt, s, c, m, a, old_lp, adv, ret):
        total = 0.0
        N = s.size(0)
        idxs = np.arange(N)
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.cfg.batch_size):
                mb = idxs[start:start + self.cfg.batch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                logits, v = agent(s[mb_t], c[mb_t], m[mb_t])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a[mb_t])
                ratio = (lp - old_lp[mb_t]).exp()
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                policy_loss = -torch.min(ratio * adv[mb_t], clipped * adv[mb_t]).mean()
                value_loss  = F.mse_loss(v, ret[mb_t])
                ent = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.cfg.max_grad_norm)
                opt.step()
                total += float(loss.item())
        return total

    # ----- inference (Algorithm 1) -------------------------------------------

    @torch.no_grad()
    def act(self, env: LHACEnv, deterministic: bool = True) -> int:
        s_t = torch.tensor(env.observe(), dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(env.per_cell_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        m_t = torch.tensor(env.feasibility_mask(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits1, _ = self.agent1(s_t, c_t, m_t)
        logits2, _ = self.agent2(s_t, c_t, m_t)
        p1 = F.softmax(logits1, dim=-1).squeeze(0)
        p2 = F.softmax(logits2, dim=-1).squeeze(0)
        a, _ = self.tlo.select(p1, p2, m_t.squeeze(0), deterministic=deterministic)
        return int(a)

    # ----- save / load -------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "agent1": self.agent1.state_dict(),
            "agent2": self.agent2.state_dict(),
            "tau": self.tlo.tau,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.agent1.load_state_dict(ck["agent1"])
        self.agent2.load_state_dict(ck["agent2"])
        self.tlo.tau = ck.get("tau", self.tlo.tau_min)
