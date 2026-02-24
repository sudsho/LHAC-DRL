"""Adaptive Thresholded Lexicographic Ordering (adaptive TLO).

Implements the candidate set
    A_hat_1(s) = { a in A(s) : pi_1(a|s) >= tau * max_{a'} pi_1(a'|s) }
and the EMA threshold update
    tau <- beta * max_a pi_1(a|s) + (1 - beta) * tau_prev
described by Eqs. (5)-(6) of the paper.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class AdaptiveTLOFilter:
    """Stateful threshold tracker; one instance per training run."""

    def __init__(
        self,
        beta: float = 0.10,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
    ):
        self.beta = float(beta)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau: float = tau_min

    def reset(self) -> None:
        self.tau = self.tau_min

    def update(self, max_p1: float) -> float:
        """EMA update on Agent 1's argmax probability."""
        new_tau = self.beta * float(max_p1) + (1.0 - self.beta) * self.tau
        self.tau = float(np.clip(new_tau, self.tau_min, self.tau_max))
        return self.tau

    # ----- candidate-set selector -----

    def candidate_mask(
        self,
        probs1: torch.Tensor,           # (n_actions,) Agent 1's masked probs
        feasible: torch.Tensor,         # (n_actions,) {0,1}
    ) -> torch.Tensor:
        """Return a {0,1} mask over A_hat_1(s).

        Falls back to feasible mask if the candidate set is empty.
        """
        max_p1 = float(probs1.max().item())
        cutoff = self.tau * max_p1
        keep = (probs1 >= cutoff).float() * feasible
        if keep.sum() < 1.0:
            keep = feasible.clone().float()
        return keep

    # ----- one-shot lexicographic selection -----

    def select(
        self,
        probs1: torch.Tensor,
        probs2: torch.Tensor,
        feasible: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Run a full lexicographic action-selection pass.

        Returns (selected action index, updated tau).
        """
        max_p1 = float((probs1 * feasible).max().item())
        self.update(max_p1)
        cand = self.candidate_mask(probs1 * feasible, feasible)
        masked2 = probs2 * cand
        if masked2.sum() <= 0:
            masked2 = cand
        masked2 = masked2 / masked2.sum()
        if deterministic:
            a = int(masked2.argmax().item())
        else:
            a = int(torch.multinomial(masked2, 1).item())
        return a, self.tau
