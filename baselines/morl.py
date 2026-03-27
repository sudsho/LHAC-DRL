"""Multi-objective RL baselines used in Section 5.6 (Figure 7).

  * PPOLagrangian   -- constrained MDP via Lagrangian relaxation
                       (Stooke, Achiam, Abbeel, 2020)
  * RCPO            -- reward-constrained policy optimisation
                       (Tessler, Mankowitz, Mannor, 2019)
  * EnvelopeMORL    -- generalised algorithm for multi-objective RL
                       (Yang, Sun, Narasimhan, 2019)
  * LPPO            -- lexicographic PPO with fixed threshold tau
                       (Zhang, Lin, Han, Lv, 2023)

Each scheduler exposes the same `.schedule(env) -> summary` interface
as the LHAC inference path so they can be benchmarked uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from lhac.env import LHACEnv
from lhac.networks import ActorCritic


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

class _SingleAgentScheduler:
    """Generic single-actor scheduler used as the inference shell for the
    four MORL baselines (each parameterises its training differently
    but uses the same inference path)."""

    def __init__(self, n_actions: int, device: Optional[str] = None):
        self.n_actions = n_actions
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------

class PPOLagrangianScheduler(_SingleAgentScheduler):
    """PPO with a Lagrangian dual variable on the tardiness constraint."""

    def __init__(self, n_actions: int, lam_init: float = 1.0, **kw):
        super().__init__(n_actions, **kw)
        self.lam = float(lam_init)


class RCPOScheduler(_SingleAgentScheduler):
    """Reward-constrained policy optimisation (penalty form)."""

    def __init__(self, n_actions: int, penalty: float = 0.5, **kw):
        super().__init__(n_actions, **kw)
        self.penalty = float(penalty)


class EnvelopeMORLScheduler(_SingleAgentScheduler):
    """Envelope MORL: vectorised value function with preference weights."""

    def __init__(self, n_actions: int, n_objectives: int = 2, **kw):
        super().__init__(n_actions, **kw)
        self.n_objectives = int(n_objectives)


class LPPOScheduler(_SingleAgentScheduler):
    """Lexicographic PPO with a fixed threshold tau (no adaptation)."""

    def __init__(self, n_actions: int, tau: float = 0.10, **kw):
        super().__init__(n_actions, **kw)
        self.tau = float(tau)
