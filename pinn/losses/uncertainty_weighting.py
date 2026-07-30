"""
Uncertainty-based multi-task loss weighting (Kendall, Gal & Cipolla, 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TASK_NAMES


class UncertaintyWeighting(nn.Module):
    """
    Learns per-task homoscedastic uncertainty sigma_i and combines task
    losses as:

        L = sum_i [ 1 / (2 * sigma_i^2) * L_i  +  log(sigma_i) ]

    sigma_i is parametrized as softplus(raw_sigma_i) + eps rather than
    exp(log_sigma_i): softplus grows ~linearly for large inputs (no
    exponential blow-up) and is bounded below by 0 (approached smoothly),
    so sigma_i > 0 is guaranteed without the runaway growth/collapse risk
    of an exp() parametrization. `init_log_sigma` is kept as the
    parameter name for backward-compatible call sites, but it initializes
    the RAW (pre-softplus) parameter.
    """

    def __init__(self, num_tasks: int = len(TASK_NAMES), init_log_sigma: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.raw_sigma = nn.Parameter(torch.full((num_tasks,), float(init_log_sigma)))

    def forward(self, losses: list):
        """
        losses: list of scalar loss tensors, length == num_tasks, in a
                FIXED order (e.g. [classification, vsg, vsl, physics,
                mass_conservation]).

        Returns:
            total_loss     : scalar tensor, sum of all weighted terms
            weighted_terms : list of scalar tensors, one weighted term per
                              task (same order as `losses`) -- useful for
                              logging each task's actual contribution to
                              the gradient.
        """
        assert len(losses) == self.num_tasks, (
            f"Expected {self.num_tasks} losses, got {len(losses)}"
        )
        sigma = F.softplus(self.raw_sigma) + 1e-6

        weighted_terms = []
        total = 0.0
        for i, loss_i in enumerate(losses):
            term = loss_i / (2.0 * sigma[i] ** 2) + torch.log(sigma[i])
            weighted_terms.append(term)
            total = total + term
        return total, weighted_terms

    def get_sigmas(self):
        with torch.no_grad():
            return (F.softplus(self.raw_sigma) + 1e-6).detach().cpu().numpy()
