"""
Multi-Task Physics-Informed Neural Network architecture.
"""

import torch
import torch.nn as nn

from ..config import (
    HIDDEN_SIZE,
    NUM_CLASSES,
    NUM_FEATURES,
    SERIES_LENGTH,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
)
from .tcn import TemporalConvNet


class MultiTaskPINN(nn.Module):
    """
    Predicts ONLY three quantities -- flow regime (classifier), superficial
    gas velocity Vsg, and superficial liquid velocity Vsl. There is NO
    auxiliary "physics_head" that learns to regress the pressure gradient
    directly from the shared trunk; that would let the network satisfy the
    physics loss by memorizing pressure -> dP/dx mappings independently of
    whether its own velocity predictions are physically consistent, which
    defeats the purpose of a physics constraint.

    Instead, the physics residual is computed OUTSIDE this module (see
    pinn.losses.physics_loss.compute_physics_loss) directly and only from
    the predicted velocities, via a fully differentiable analytical chain:
    Vm -> alpha -> rho_mix -> Re -> friction factor (Blasius) ->
    Darcy-Weisbach dP/dx. Gradients from that residual therefore propagate
    back through real physics equations into the velocity head -- and
    nowhere else -- which is the defining property of a PINN constraint.
    """

    def __init__(
        self,
        window_size: int = SERIES_LENGTH,
        num_features: int = NUM_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = NUM_CLASSES,
        cls_dropout: float = 0.4,
        reg_dropout: float = 0.15,
        tcn_channels: list = None,
        tcn_kernel_size: int = TCN_KERNEL_SIZE,
    ):
        super().__init__()

        tcn_channels = tcn_channels if tcn_channels is not None else TCN_CHANNELS

        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=reg_dropout,   # lighter -- this branch feeds BOTH heads
        )
        tcn_out_dim = tcn_channels[-1]

        self.fc_feats = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(reg_dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Shared trunk.
        self.shared = nn.Sequential(
            nn.Linear(tcn_out_dim + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(reg_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Classification head -- more aggressive dropout than the
        # regression/physics-facing branch.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Velocity head: LayerNorm (not BatchNorm1d) so it has no
        # running-statistics train/eval mismatch and works correctly even
        # for batch size 1 (e.g. single-sample inference). This is the
        # physics-facing head -- gradients propagating back from the
        # analytical physics residual flow through it directly.
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(reg_dropout),
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2),   # [Vsg_scaled, Vsl_scaled]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # The TCN initializes its own conv weights (see TemporalBlock,
        # following the standard locuslab/TCN convention for
        # weight-normalized convs) -- skip it here.
        for name, m in self.named_modules():
            if name.startswith("tcn."):
                continue
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pressure_window: torch.Tensor, features: torch.Tensor):
        tcn_out = self.tcn(pressure_window.unsqueeze(1))   # (B, C, L)
        x = tcn_out[:, :, -1]                               # last timestep: causal, so this
                                                              # already summarizes the whole
                                                              # trace given a full-coverage RF
        f = self.fc_feats(features)
        shared = self.shared(torch.cat([x, f], dim=1))
        # Only two outputs: classification logits and velocity predictions
        # [Vsg, Vsl]. The physics residual is computed downstream,
        # analytically, from `velocity_pred`.
        return (
            self.classifier(shared),
            self.velocity_head(shared),
        )
