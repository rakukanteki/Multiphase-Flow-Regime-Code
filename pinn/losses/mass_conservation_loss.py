"""
Mass-conservation residual (steady, incompressible mixture continuity).
"""

import torch
import torch.nn.functional as F

from ..config import PIPE_AREA


def compute_mass_conservation_loss(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    scaler_vsg,
    scaler_vsl,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
) -> tuple:
    """
    Fully differentiable mass-conservation residual for a steady,
    incompressible mixture in a fixed cross-section pipe.

    Starting from the continuity equation
        d(rho*A)/dt + d(rho*u*A)/dx = 0
    steady-state (d/dt = 0) and incompressible + constant A collapse this
    to d(rho*u*A)/dx = 0, i.e. the volumetric mixture flow rate

        Q = A * (Vsg + Vsl)

    must be constant along the pipe, and must equal the file's actual
    (set-point) total flow rate. This is computed from BOTH the model's
    predicted velocities (`velocity_pred`, in-graph, gradients flow back
    into the velocity head) and the true per-file set-point velocities
    (`velocity_target`, detached ground truth -- never used as a model
    input, only as the conservation anchor).

    Both sides are inverse-scaled back to physical units (m/s) before
    forming Q, then whitened using q_mean/q_std (computed once from the
    TRAINING split's target Q only) so this residual sits on a comparable
    scale to the other task losses.

    Returns:
        loss_mass : scalar MSE tensor (keeps grad for backprop)
        q_pred    : detached (N,1) tensor, RAW (unnormalized, m^3/s)
        q_target  : detached (N,1) tensor, RAW (unnormalized, m^3/s)
    """
    eps = 1e-8

    vsg_center = torch.tensor(scaler_vsg.center_[0], device=velocity_pred.device, dtype=torch.float32)
    vsg_scale = torch.tensor(scaler_vsg.scale_[0], device=velocity_pred.device, dtype=torch.float32)
    vsl_center = torch.tensor(scaler_vsl.center_[0], device=velocity_pred.device, dtype=torch.float32)
    vsl_scale = torch.tensor(scaler_vsl.scale_[0], device=velocity_pred.device, dtype=torch.float32)

    # Predicted side -- in-graph inverse scaling, gradients flow back into
    # the velocity head (same pattern as compute_physics_loss).
    vsg_pred_phys = (velocity_pred[:, 0:1] * vsg_scale) + vsg_center
    vsl_pred_phys = (velocity_pred[:, 1:2] * vsl_scale) + vsl_center
    vsg_pred_phys = torch.clamp(vsg_pred_phys, min=1e-4)
    vsl_pred_phys = torch.clamp(vsl_pred_phys, min=1e-4)

    # Target side -- the true set-point velocities for this file, detached
    # (this is the conservation ANCHOR, not a model output).
    vsg_tgt_phys = ((velocity_target[:, 0:1] * vsg_scale) + vsg_center).detach()
    vsl_tgt_phys = ((velocity_target[:, 1:2] * vsl_scale) + vsl_center).detach()

    q_pred = PIPE_AREA * (vsg_pred_phys + vsl_pred_phys)
    q_target = (PIPE_AREA * (vsg_tgt_phys + vsl_tgt_phys)).detach()

    q_pred_norm = (q_pred - q_mean) / (q_std + eps)
    q_target_norm = (q_target - q_mean) / (q_std + eps)

    loss_mass = F.mse_loss(q_pred_norm, q_target_norm)

    return loss_mass, q_pred.detach(), q_target.detach()
