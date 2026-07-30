"""
Evaluation utilities -- runs a trained model over a held-out loader and
collects original-scale predictions/targets/physics-consistency numbers.

No plotting, no print statements.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from ..config import DEVICE
from ..losses import compute_mass_conservation_loss, compute_physics_loss


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    scaler_vsg,
    scaler_vsl,
    dpdx_mean: torch.Tensor,
    dpdx_std: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
) -> dict:
    """
    Run the model over a held-out loader (validation or test) and collect
    original-scale predictions/targets for reporting.

    The physics-consistency quantity returned here ("phys_pred") is NOT an
    auxiliary head output -- it is the same analytical Darcy-Weisbach
    dP/dx computed by `compute_physics_loss` (drift-flux gas holdup),
    evaluated on this model's predicted velocities: "if we trusted this
    model's Vsg/Vsl predictions and pushed them through real pipe-flow
    physics, how close is the resulting pressure gradient to what was
    actually measured?"

    Likewise, "q_pred"/"q_target" are the mass-conservation quantity
    Q = A*(Vsg + Vsl) computed by `compute_mass_conservation_loss` from
    the predicted vs. actual set-point velocities of this loader's files.
    """
    model.eval()
    preds_all, labels_all = [], []
    vel_pred_scaled, vel_tgt_scaled = [], []
    phys_pred_all, phys_meas_all = [], []
    q_pred_all, q_tgt_all = [], []

    with torch.no_grad():
        for batch in loader:
            pressure = batch["pressure_window"].to(DEVICE)
            feats = batch["features"].to(DEVICE)
            lbl = batch["label"].to(DEVICE)
            dp_dx = batch["dp_dx_measured"].to(DEVICE)
            vel_tgt = torch.cat([batch["vsg"].to(DEVICE), batch["vsl"].to(DEVICE)], dim=1)

            class_logits, vel_pred = model(pressure, feats)
            _, preds = torch.max(class_logits, 1)

            # Reuse the exact same differentiable equations used during
            # training for consistency between train-time and report-time
            # physics (gradients not needed here since we're in no_grad).
            _, dp_dx_from_vel = compute_physics_loss(
                vel_pred, dp_dx, scaler_vsg, scaler_vsl, dpdx_mean, dpdx_std
            )
            _, q_pred, q_tgt = compute_mass_conservation_loss(
                vel_pred, vel_tgt, scaler_vsg, scaler_vsl, q_mean, q_std
            )

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
            vel_pred_scaled.extend(vel_pred.cpu().numpy())
            vel_tgt_scaled.extend(vel_tgt.cpu().numpy())
            phys_pred_all.extend(dp_dx_from_vel.cpu().numpy().reshape(-1))
            phys_meas_all.extend(dp_dx.cpu().numpy().reshape(-1))
            q_pred_all.extend(q_pred.cpu().numpy().reshape(-1))
            q_tgt_all.extend(q_tgt.cpu().numpy().reshape(-1))

    vel_pred_scaled = np.array(vel_pred_scaled)
    vel_tgt_scaled = np.array(vel_tgt_scaled)

    vsg_pred_orig = scaler_vsg.inverse_transform(vel_pred_scaled[:, 0:1])
    vsg_tgt_orig = scaler_vsg.inverse_transform(vel_tgt_scaled[:, 0:1])
    vsl_pred_orig = scaler_vsl.inverse_transform(vel_pred_scaled[:, 1:2])
    vsl_tgt_orig = scaler_vsl.inverse_transform(vel_tgt_scaled[:, 1:2])

    return {
        "predictions": np.array(preds_all),
        "true_labels": np.array(labels_all),
        "vel_pred": np.hstack([vsg_pred_orig, vsl_pred_orig]),
        "vel_target": np.hstack([vsg_tgt_orig, vsl_tgt_orig]),
        "phys_pred": np.array(phys_pred_all),
        "phys_measured": np.array(phys_meas_all),
        "q_pred": np.array(q_pred_all),
        "q_target": np.array(q_tgt_all),
    }
