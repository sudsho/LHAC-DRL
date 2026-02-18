"""Actor-critic networks for LHAC.

Two independent actor-critic pairs (one per objective). Both share the
same compact 78-D state and a CASE cross-attention head that consumes a
variable-length sequence of per-cell features (Section 4.2.2 of the
paper, Eq. 7).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CASE cross-attention encoder
# ---------------------------------------------------------------------------

class CASEEncoder(nn.Module):
    """Multi-head cross-attention over per-cell features.

    Server-derived query attends to per-cell key/value projections; the
    attended embedding h_CASE is concatenated with the global state.
    """

    def __init__(
        self,
        state_dim: int = 78,
        cell_feat_dim: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # state -> server query projection
        self.state_proj = nn.Linear(state_dim, d_model)

        # cell -> key, value projections
        self.k_proj = nn.Linear(cell_feat_dim, d_model)
        self.v_proj = nn.Linear(cell_feat_dim, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        state: torch.Tensor,             # (B, state_dim)
        cell_feats: torch.Tensor,        # (B, n_cells, cell_feat_dim)
    ) -> torch.Tensor:
        B, n_cells, _ = cell_feats.shape

        # query
        s_proj = F.relu(self.state_proj(state))              # (B, d_model)
        q = self.q_proj(s_proj).unsqueeze(1)                 # (B, 1, d_model)

        # key, value
        k = self.k_proj(cell_feats)                          # (B, n_cells, d_model)
        v = self.v_proj(cell_feats)                          # (B, n_cells, d_model)

        # split heads
        def split(t):
            return t.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        qh = split(q)        # (B, h, 1, d_head)
        kh = split(k)        # (B, h, n_cells, d_head)
        vh = split(v)        # (B, h, n_cells, d_head)

        attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(attn, dim=-1)                    # (B, h, 1, n_cells)
        head_out = weights @ vh                              # (B, h, 1, d_head)
        merged = head_out.transpose(1, 2).reshape(B, 1, self.d_model).squeeze(1)
        h_case = self.norm(s_proj + self.out_proj(merged))   # residual + LN
        return h_case                                         # (B, d_model)


# ---------------------------------------------------------------------------
# Actor-critic head
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Single actor-critic for one objective.

    Inputs:
        state       (B, state_dim)
        cell_feats  (B, n_cells, cell_feat_dim)
        mask        (B, n_actions)        bool/float feasibility mask
    """

    def __init__(
        self,
        state_dim: int = 78,
        cell_feat_dim: int = 12,
        n_actions: int = 55,
        d_model: int = 128,
        hidden: int = 256,
    ):
        super().__init__()
        self.encoder = CASEEncoder(state_dim, cell_feat_dim, d_model)

        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        cell_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(state, cell_feats)
        logits = self.actor(h)
        # large negative bias on infeasible actions
        logits = logits + (mask + 1e-9).log()
        value = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def policy_probs(
        self,
        state: torch.Tensor,
        cell_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(state, cell_feats, mask)
        return F.softmax(logits, dim=-1)
