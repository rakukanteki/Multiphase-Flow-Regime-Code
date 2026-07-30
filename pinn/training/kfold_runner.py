"""
K-fold training orchestration.

Consumes a list of pre-built `FoldBundle`s (one per fold) -- each bundle
carries the fold's own DataLoaders, fitted scalers, and normalization
statistics. Building those (reading files, fitting scalers, constructing
Datasets/DataLoaders) is deliberately left to your own data layer; this
module owns only the cross-validation *training* orchestration:
model/optimizer/scheduler construction per fold, delegating to
`train_model`, and selecting the best fold by validation loss.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import (
    DEVICE,
    LEARNING_RATE,
    NUM_CLASSES,
    TASK_NAMES,
    UNCERTAINTY_LEARNING_RATE,
    WEIGHT_DECAY,
)
from ..losses import UncertaintyWeighting
from ..models import MultiTaskPINN
from .trainer import train_model


@dataclass
class FoldBundle:
    """Everything one fold of training needs, prepared by the caller's
    own data layer."""
    train_loader: DataLoader
    val_loader: DataLoader
    scaler_vsg: object
    scaler_vsl: object
    dpdx_mean: torch.Tensor
    dpdx_std: torch.Tensor
    q_mean: torch.Tensor
    q_std: torch.Tensor
    labels_train: np.ndarray   # class index per training sample, for class weights


def _build_model_and_optimizer(learning_rate: float, weight_decay: float,
                                uncertainty_lr: float):
    model = MultiTaskPINN().to(DEVICE)
    uncertainty_weighting = UncertaintyWeighting(num_tasks=len(TASK_NAMES)).to(DEVICE)

    optimizer = optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                # The sigma_i parameters directly rescale every task's
                # gradient magnitude, so if they move as fast as the model
                # weights they can chase noisy per-batch loss values and
                # destabilize the very weighting they're supposed to
                # stabilize. A learning rate an order of magnitude smaller
                # lets sigma_i track the *slow-moving* relative difficulty
                # of each task instead of each batch's noise. No weight
                # decay: regularizing sigma_i toward 0 would fight the
                # mechanism itself.
                "params": uncertainty_weighting.parameters(),
                "lr": uncertainty_lr,
                "weight_decay": 0.0,
            },
        ],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    return model, uncertainty_weighting, optimizer, scheduler


def run_kfold_train_val_test(
    fold_bundles: List[FoldBundle],
    models_dir: Path,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    uncertainty_lr: float = UNCERTAINTY_LEARNING_RATE,
) -> dict:
    """
    Trains one model per fold in `fold_bundles`, then returns the
    checkpoint/scalers/history of the fold with the lowest best validation
    loss, plus every fold's training history for downstream reporting.

    Returns a dict with:
        all_histories   : list[dict], one train_model() history per fold
        best_fold_idx   : int
        best_ckpt_path  : Path to the best fold's checkpoint
        best_scalers    : dict (scaler_vsg, scaler_vsl, dpdx_mean/std, q_mean/std)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_histories = []
    fold_scalers = []
    fold_ckpt_paths = []

    best_fold_idx: Optional[int] = None
    best_fold_val_loss = float("inf")

    criterion_vel = nn.SmoothL1Loss(beta=0.1)

    for fold_idx, bundle in enumerate(fold_bundles):
        cls_counts = np.bincount(bundle.labels_train, minlength=NUM_CLASSES)
        cls_weights = torch.FloatTensor(
            (1.0 / (cls_counts + 1e-8)) / (1.0 / (cls_counts + 1e-8)).sum() * NUM_CLASSES
        ).to(DEVICE)
        criterion_class = nn.CrossEntropyLoss(weight=cls_weights)

        model, uncertainty_weighting, optimizer, scheduler = _build_model_and_optimizer(
            learning_rate, weight_decay, uncertainty_lr
        )

        history = train_model(
            model, uncertainty_weighting,
            bundle.train_loader, bundle.val_loader,
            criterion_class, criterion_vel,
            optimizer, scheduler,
            bundle.scaler_vsg, bundle.scaler_vsl,
            bundle.dpdx_mean, bundle.dpdx_std,
            bundle.q_mean, bundle.q_std,
            models_dir=models_dir,
            ckpt_name=f"best_pinn_model_fold{fold_idx + 1}_ckpt.pth",
        )

        all_histories.append(history)
        fold_scalers.append({
            "vsg": bundle.scaler_vsg,
            "vsl": bundle.scaler_vsl,
            "dpdx_mean": bundle.dpdx_mean,
            "dpdx_std": bundle.dpdx_std,
            "q_mean": bundle.q_mean,
            "q_std": bundle.q_std,
        })
        fold_ckpt_paths.append(history["ckpt_path"])

        if history["best_val_loss"] < best_fold_val_loss:
            best_fold_val_loss = history["best_val_loss"]
            best_fold_idx = fold_idx

    return {
        "all_histories": all_histories,
        "best_fold_idx": best_fold_idx,
        "best_ckpt_path": fold_ckpt_paths[best_fold_idx],
        "best_scalers": fold_scalers[best_fold_idx],
    }
