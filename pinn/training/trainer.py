"""
Per-fold epoch training loop.

Accepts already-built DataLoaders (dependency injection) -- this module
has no knowledge of how samples were read from disk.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import DEVICE, EARLY_STOP_PATIENCE, EPOCHS
from ..losses import UncertaintyWeighting, compute_total_loss


def train_model(
    model: nn.Module,
    uncertainty_weighting: UncertaintyWeighting,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion_class: nn.Module,
    criterion_vel: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler_vsg,
    scaler_vsl,
    dpdx_mean: torch.Tensor,
    dpdx_std: torch.Tensor,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
    models_dir: Path,
    ckpt_name: str = "best_pinn_model_ckpt.pth",
    epochs: int = EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
) -> dict:
    """
    Trains `model` + `uncertainty_weighting` for up to `epochs` epochs,
    with early stopping on validation total loss and checkpointing of the
    best epoch. Returns a history dict of per-epoch metrics plus the best
    validation loss/accuracy and the checkpoint path.
    """
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_vsg_loss": [], "val_vsg_loss": [],
        "train_vsl_loss": [], "val_vsl_loss": [],
        "train_phys_loss": [], "val_phys_loss": [],
        "train_mass_loss": [], "val_mass_loss": [],
        "val_vel_mae": [],
        # weighted (uncertainty-weighted) contributions, for PINN-influence analysis
        "train_contrib_class": [], "train_contrib_vsg": [],
        "train_contrib_vsl": [], "train_contrib_phys": [], "train_contrib_mass": [],
        # learned task sigmas, tracked per epoch for diagnostics
        "sigma_class": [], "sigma_vsg": [], "sigma_vsl": [], "sigma_phys": [], "sigma_mass": [],
    }

    best_val_loss = float("inf")
    best_val_acc_at_best_loss = 0.0
    epochs_no_improve = 0
    best_ckpt_path = Path(models_dir) / ckpt_name

    for epoch in range(1, epochs + 1):

        # ---------------- Training ----------------
        model.train()
        uncertainty_weighting.train()
        t_loss = t_vsg_loss = t_vsl_loss = t_phys_loss = t_mass_loss = 0.0
        t_contrib_c = t_contrib_vsg = t_contrib_vsl = t_contrib_p = t_contrib_m = 0.0
        t_correct = t_total = 0

        for batch in train_loader:
            if batch["label"].size(0) <= 1:
                continue

            pressure = batch["pressure_window"].to(DEVICE)
            feats = batch["features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            dp_dx = batch["dp_dx_measured"].to(DEVICE)
            vel_tgt = torch.cat([batch["vsg"].to(DEVICE), batch["vsl"].to(DEVICE)], dim=1)

            optimizer.zero_grad()
            class_logits, vel_pred = model(pressure, feats)
            (total, lc, lvsg, lvsl, lp, lm, _, _, _,
             wc, wvsg, wvsl, wp, wm) = compute_total_loss(
                class_logits, vel_pred,
                labels, vel_tgt, dp_dx,
                criterion_class, criterion_vel,
                scaler_vsg, scaler_vsl,
                uncertainty_weighting,
                dpdx_mean, dpdx_std,
                q_mean, q_std,
            )
            total.backward()
            # Clip the model and the uncertainty-weighting parameters
            # SEPARATELY (rather than as one combined pool). The sigma_i
            # parameters live on their own, much smaller scale than the
            # network weights, so a single shared clip norm either barely
            # touches them or over-clips the model; a tighter, dedicated
            # norm keeps their updates gentle without disturbing the
            # model's gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(uncertainty_weighting.parameters(), max_norm=0.5)
            optimizer.step()

            t_loss += total.item()
            t_vsg_loss += lvsg.item()
            t_vsl_loss += lvsl.item()
            t_phys_loss += lp.item()
            t_mass_loss += lm.item()
            t_contrib_c += wc.item()
            t_contrib_vsg += wvsg.item()
            t_contrib_vsl += wvsl.item()
            t_contrib_p += wp.item()
            t_contrib_m += wm.item()
            _, preds = torch.max(class_logits, 1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()

        n_train = max(len(train_loader), 1)
        train_acc = 100.0 * t_correct / t_total if t_total > 0 else 0.0
        avg_t = t_loss / n_train
        avg_tvsg = t_vsg_loss / n_train
        avg_tvsl = t_vsl_loss / n_train
        avg_tp = t_phys_loss / n_train
        avg_tm = t_mass_loss / n_train

        # ---------------- Validation ----------------
        model.eval()
        uncertainty_weighting.eval()
        v_loss = v_vsg_loss = v_vsl_loss = v_phys_loss = v_mass_loss = 0.0
        v_correct = v_total = 0
        vel_mae_sum = vel_count = 0

        with torch.no_grad():
            for batch in val_loader:
                pressure = batch["pressure_window"].to(DEVICE)
                feats = batch["features"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                dp_dx = batch["dp_dx_measured"].to(DEVICE)
                vel_tgt = torch.cat([batch["vsg"].to(DEVICE), batch["vsl"].to(DEVICE)], dim=1)

                class_logits, vel_pred = model(pressure, feats)
                (total, lc, lvsg, lvsl, lp, lm, _, _, _,
                 wc, wvsg, wvsl, wp, wm) = compute_total_loss(
                    class_logits, vel_pred,
                    labels, vel_tgt, dp_dx,
                    criterion_class, criterion_vel,
                    scaler_vsg, scaler_vsl,
                    uncertainty_weighting,
                    dpdx_mean, dpdx_std,
                    q_mean, q_std,
                )

                v_loss += total.item()
                v_vsg_loss += lvsg.item()
                v_vsl_loss += lvsl.item()
                v_phys_loss += lp.item()
                v_mass_loss += lm.item()
                _, preds = torch.max(class_logits, 1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()
                mae = torch.mean(torch.abs(vel_pred - vel_tgt)).item()
                vel_mae_sum += mae * labels.size(0)
                vel_count += labels.size(0)

        n_val = max(len(val_loader), 1)
        val_acc = 100.0 * v_correct / v_total if v_total > 0 else 0.0
        avg_v = v_loss / n_val
        avg_vvsg = v_vsg_loss / n_val
        avg_vvsl = v_vsl_loss / n_val
        avg_vp = v_phys_loss / n_val
        avg_vm = v_mass_loss / n_val
        avg_mae = vel_mae_sum / vel_count if vel_count > 0 else 0.0

        scheduler.step(avg_v)

        sigmas = uncertainty_weighting.get_sigmas()

        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_vsg_loss"].append(avg_tvsg)
        history["val_vsg_loss"].append(avg_vvsg)
        history["train_vsl_loss"].append(avg_tvsl)
        history["val_vsl_loss"].append(avg_vvsl)
        history["train_phys_loss"].append(avg_tp)
        history["val_phys_loss"].append(avg_vp)
        history["train_mass_loss"].append(avg_tm)
        history["val_mass_loss"].append(avg_vm)
        history["val_vel_mae"].append(avg_mae)
        history["train_contrib_class"].append(t_contrib_c / n_train)
        history["train_contrib_vsg"].append(t_contrib_vsg / n_train)
        history["train_contrib_vsl"].append(t_contrib_vsl / n_train)
        history["train_contrib_phys"].append(t_contrib_p / n_train)
        history["train_contrib_mass"].append(t_contrib_m / n_train)
        history["sigma_class"].append(float(sigmas[0]))
        history["sigma_vsg"].append(float(sigmas[1]))
        history["sigma_vsl"].append(float(sigmas[2]))
        history["sigma_phys"].append(float(sigmas[3]))
        history["sigma_mass"].append(float(sigmas[4]))

        # Select checkpoint by TOTAL validation loss (balances all five tasks).
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_val_acc_at_best_loss = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "uncertainty_weighting_state_dict": uncertainty_weighting.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": avg_v,
                    "val_mae": avg_mae,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            break

    history["best_val_acc"] = best_val_acc_at_best_loss
    history["best_val_loss"] = best_val_loss
    history["ckpt_path"] = best_ckpt_path
    return history
