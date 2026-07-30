"""
Combined multi-task loss: classification + Vsg + Vsl + physics +
mass-conservation, weighted via learned uncertainty weighting.
"""

import torch
import torch.nn as nn

from .physics_loss import compute_physics_loss
from .mass_conservation_loss import compute_mass_conservation_loss
from .uncertainty_weighting import UncertaintyWeighting


def compute_total_loss(
    class_logits: torch.Tensor,
    velocity_pred: torch.Tensor,
    labels: torch.Tensor,
    velocity_target: torch.Tensor,
    dp_dx_measured: torch.Tensor,
    criterion_class: nn.Module,
    criterion_vel: nn.Module,
    scaler_vsg,
    scaler_vsl,
    uncertainty_weighting: UncertaintyWeighting,
    dpdx_mean: torch.Tensor,
    dpdx_std: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
) -> tuple:
    """
    Computes the FIVE raw task losses -- classification, Vsg regression,
    Vsl regression, Darcy-Weisbach physics residual (drift-flux), and
    mass-conservation residual -- and combines them via the learned
    uncertainty weights instead of fixed lambda scalars.

    Vsg and Vsl losses are kept SEPARATE (rather than one combined
    Huber/SmoothL1 over the concatenated [Vsg, Vsl] vector) because their
    physical ranges differ; a single combined loss would let the
    larger-range target dominate the gradient of the smaller-range one.
    Each still gets its own learned uncertainty weight, so the network can
    further balance them automatically during training.

    Returns:
        total, loss_class, loss_vsg, loss_vsl, loss_phys, loss_mass,
        dp_dx_from_vel, q_pred, q_target,
        weighted_class, weighted_vsg, weighted_vsl, weighted_phys, weighted_mass
    """
    loss_class = criterion_class(class_logits, labels)
    loss_vsg = criterion_vel(velocity_pred[:, 0:1], velocity_target[:, 0:1])
    loss_vsl = criterion_vel(velocity_pred[:, 1:2], velocity_target[:, 1:2])
    loss_phys, dp_dx_from_vel = compute_physics_loss(
        velocity_pred, dp_dx_measured, scaler_vsg, scaler_vsl, dpdx_mean, dpdx_std
    )
    loss_mass, q_pred, q_target = compute_mass_conservation_loss(
        velocity_pred, velocity_target, scaler_vsg, scaler_vsl, q_mean, q_std
    )

    total, weighted_terms = uncertainty_weighting(
        [loss_class, loss_vsg, loss_vsl, loss_phys, loss_mass]
    )
    weighted_class, weighted_vsg, weighted_vsl, weighted_phys, weighted_mass = weighted_terms

    return (
        total, loss_class, loss_vsg, loss_vsl, loss_phys, loss_mass,
        dp_dx_from_vel, q_pred, q_target,
        weighted_class, weighted_vsg, weighted_vsl, weighted_phys, weighted_mass,
    )
