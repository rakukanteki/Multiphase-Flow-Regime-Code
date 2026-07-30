"""
Physical constants, closure coefficients, and training hyperparameters.

Deliberately excludes dataset location / folder constants (BASE_DIR,
SUB_FOLDERS) -- those belong to your own data-loading layer.
"""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Task / class metadata
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Dispersed Flow", "Plug Flow", "Slug Flow"]
NUM_CLASSES = 3

# Fixed task ordering used everywhere in losses/training -- keep in sync
# with UncertaintyWeighting(num_tasks=len(TASK_NAMES)).
TASK_NAMES = ["classification", "vsg", "vsl", "physics", "mass_conservation"]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
PIPE_DIAMETER = 2 * 0.0254          # metres  (2-inch pipe)
PIPE_LENGTH = 7.25                  # metres
PIPE_AREA = float(np.pi / 4.0 * PIPE_DIAMETER ** 2)   # m^2, cross-sectional area
RHO_WATER = 998.0                   # kg/m3
RHO_AIR_MEAN = 1.49                 # kg/m3
P_ATM_BAR = 1.01325                 # bar
GRAVITY = 9.81                      # m/s^2

# Dynamic viscosities (Pa*s) for the homogeneous-mixture viscosity used in
# the Reynolds number below (water and air near ambient rig conditions).
MU_WATER = 1.002e-3    # Pa.s
MU_AIR = 1.81e-5       # Pa.s

# Reynolds number at which flow transitions from laminar to turbulent for
# friction-factor purposes (classical Re ~ 2300 threshold; not a fitted
# constant). Kept as a documented reference -- the friction factor below
# uses the turbulent (Blasius) branch unconditionally since this rig's
# operating envelope is turbulent throughout.
RE_TRANSITION = 2300.0

# ---------------------------------------------------------------------------
# Drift-flux closure (Zuber-Findlay form, Bendiksen 1984 horizontal/
# near-horizontal coefficients)
# ---------------------------------------------------------------------------
# Slug and plug flow are NOT homogeneous (no-slip) regimes: the gas phase
# moves faster than the mixture average because it migrates toward the pipe
# centre/top of each liquid slug. The homogeneous model (alpha = Vsg / Vm)
# is replaced by the drift-flux model:
#
#       Vg   = C0 * Vm + V_gj        (actual, in-situ gas-phase velocity)
#       Vsg  = alpha * Vg            (definition of superficial velocity)
#   =>  alpha = Vsg / (C0 * Vm + V_gj)
#
# C0 is the distribution parameter and V_gj is the buoyancy-driven drift
# velocity. Both are fixed, physically-motivated closure constants (not
# fitted to any dataset). Setting C0=1, V_gj=0 recovers the homogeneous
# model exactly, so this is a strict generalization of it.
DRIFT_C0 = 1.05
DRIFT_VGJ_COEFF = 0.35
DRIFT_VGJ = float(DRIFT_VGJ_COEFF * np.sqrt(GRAVITY * PIPE_DIAMETER))   # m/s

# ---------------------------------------------------------------------------
# Split hyperparameters
# ---------------------------------------------------------------------------
TEST_FRAC = 0.15       # held-out, untouched final test fraction (per class)
N_SPLITS = 5            # number of cross-validation folds
SPLIT_SEED = 42         # deterministic shuffle -- same split every run

# Every file's raw pressure trace is resampled (linear interpolation) onto
# this many points before being fed into the TCN branch, since files have
# different lengths and the model needs a fixed-size input.
SERIES_LENGTH = 220

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128

# TCN branch. Dilations double every layer (1,2,4,8,16,32,64) so the
# receptive field comfortably covers the full SERIES_LENGTH sample window.
TCN_CHANNELS = [32, 32, 64, 64, 64, 64, 64]
TCN_KERNEL_SIZE = 3
NUM_FEATURES = 8   # must match features.extract_pressure_features output

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 250
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

# The learned uncertainty-weighting sigma_i parameters get their own, much
# smaller learning rate than the network so they track the slow-moving
# relative difficulty of each task rather than per-batch noise.
UNCERTAINTY_LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 70     # epochs with no val-loss improvement before stopping

# ---------------------------------------------------------------------------
# NOTE on multi-task loss weighting
# ---------------------------------------------------------------------------
# Task weighting is learned automatically during training via homoscedastic
# uncertainty weighting (Kendall, Gal & Cipolla, 2018):
#
#       L = sum_i [ 1 / (2 * sigma_i^2) * L_i  +  log(sigma_i) ]
#
# See pinn.losses.uncertainty_weighting.UncertaintyWeighting.
