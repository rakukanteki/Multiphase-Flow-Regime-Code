"""
Physics-informed loss, grounded in real measured pressure (not circular).
"""

import torch
import torch.nn.functional as F

from ..config import (
    DRIFT_C0,
    DRIFT_VGJ,
    MU_AIR,
    MU_WATER,
    PIPE_DIAMETER,
    RHO_AIR_MEAN,
    RHO_WATER,
)


def compute_physics_loss(
    velocity_pred: torch.Tensor,
    dp_dx_measured: torch.Tensor,
    scaler_vsg,
    scaler_vsl,
    dpdx_mean: torch.Tensor,
    dpdx_std: torch.Tensor,
) -> tuple:
    """
    Fully differentiable analytical physics residual, computed ONLY from
    the model's predicted velocities `velocity_pred` (never from ground
    truth Vsg/Vsl -- those are only ever used as regression *targets*
    elsewhere, never as inputs to this function).

    Chain of equations (all differentiable w.r.t. velocity_pred):

        1. Inverse-scale predictions back to physical units (m/s), keeping
           the autograd graph intact (no .detach()).
        2. Mixture velocity   Vm    = Vsg + Vsl
           Gas holdup (DRIFT-FLUX, Zuber-Findlay / Bendiksen 1984):
               alpha = Vsg / (C0 * Vm + V_gj)
           -- NOT the homogeneous, no-slip alpha = Vsg / Vm. Slug and plug
           flow are not homogeneous: the gas phase slips relative to the
           mixture, so the no-slip model systematically mis-estimates
           holdup (and hence rho_mix / Re / friction) in exactly the
           regimes this project targets. C0 and V_gj are fixed closure
           constants, not fitted.
           Mixture density    rho_mix = alpha*rho_air + (1-alpha)*rho_water
           Mixture viscosity  mu_mix  = alpha*mu_air  + (1-alpha)*mu_water
        3. Reynolds number    Re = rho_mix * Vm * D / mu_mix
        4. Friction factor (Blasius correlation -- flow is turbulent
           throughout this rig's operating envelope):
               f = 0.3164 / Re^0.25
        5. Darcy-Weisbach pressure gradient:
               dP/dx = f * rho_mix * Vm^2 / (2 * D)         [Pa/m -> bar/m]

    The residual loss is the MSE between this analytically-derived dP/dx
    and the measured dP/dx anchor, with BOTH sides normalized using
    dpdx_mean/dpdx_std (computed once from the TRAINING split's measured
    dP/dx only) so the physics residual is on a comparable, whitened scale
    rather than raw bar/m units, and no validation/test statistics ever
    leak into the normalization. Because every step is a differentiable
    torch operation, gradients flow: physics_loss -> dP/dx_pred -> f -> Re
    -> Vm/alpha -> velocity_pred -> velocity_head weights. This is what
    makes the constraint act on the network's velocity predictions rather
    than being satisfiable by an independent auxiliary head.

    Returns:
        loss_phys      : scalar MSE tensor (keeps grad for backprop)
        dp_dx_from_vel : detached (N,1) tensor, RAW (unnormalized, bar/m)
                         for logging/reporting only
    """
    eps = 1e-8

    vsg_center = torch.tensor(scaler_vsg.center_[0], device=velocity_pred.device, dtype=torch.float32)
    vsg_scale = torch.tensor(scaler_vsg.scale_[0], device=velocity_pred.device, dtype=torch.float32)
    vsl_center = torch.tensor(scaler_vsl.center_[0], device=velocity_pred.device, dtype=torch.float32)
    vsl_scale = torch.tensor(scaler_vsl.scale_[0], device=velocity_pred.device, dtype=torch.float32)

    # In-graph inverse scaling (keeps autograd chain intact -- no detach)
    vsg_phys = (velocity_pred[:, 0:1] * vsg_scale) + vsg_center
    vsl_phys = (velocity_pred[:, 1:2] * vsl_scale) + vsl_center

    # Physical velocities cannot be negative; clamp (differentiable, zero
    # sub-gradient only in the invalid region) purely for numerical
    # stability of Re / f early in training when predictions are noisy.
    vsg_phys = torch.clamp(vsg_phys, min=1e-4)
    vsl_phys = torch.clamp(vsl_phys, min=1e-4)

    # --- Mixture kinematics / properties (DRIFT-FLUX model) ---
    U_m = vsg_phys + vsl_phys
    alpha = vsg_phys / (DRIFT_C0 * U_m + DRIFT_VGJ + eps)
    alpha = torch.clamp(alpha, min=1e-4, max=1.0 - 1e-4)
    rho_mix = alpha * RHO_AIR_MEAN + (1.0 - alpha) * RHO_WATER
    mu_mix = alpha * MU_AIR + (1.0 - alpha) * MU_WATER

    # --- Reynolds number ---
    Re = rho_mix * U_m * PIPE_DIAMETER / (mu_mix + eps)

    # --- Friction factor: Blasius correlation, applied unconditionally
    #     (Re is turbulent throughout this rig's operating envelope) ---
    friction_factor = 0.3164 * torch.pow(Re + eps, -0.25)

    # --- Darcy-Weisbach, Pa/m -> bar/m (1 / 100,000), matching the units
    #     of dp_dx_measured (already in bar/m) ---
    dp_dx_from_vel = (friction_factor * rho_mix * (U_m ** 2) / (2.0 * PIPE_DIAMETER)) / 100000.0

    dp_dx_measured = dp_dx_measured.view(-1, 1)

    # --- Normalize both sides with TRAIN-set dP/dx statistics before the
    #     MSE, so the physics residual sits on a comparable, whitened scale
    #     instead of raw bar/m units. ---
    dp_dx_measured_norm = (dp_dx_measured - dpdx_mean) / (dpdx_std + eps)
    dp_dx_pred_norm = (dp_dx_from_vel - dpdx_mean) / (dpdx_std + eps)

    loss_phys = F.mse_loss(dp_dx_pred_norm, dp_dx_measured_norm)

    return loss_phys, dp_dx_from_vel.detach()
