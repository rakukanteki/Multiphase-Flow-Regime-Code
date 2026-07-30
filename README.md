# Multiphase Flow PINN — Modular Codebase

Refactored from the original notebooks into two importable packages.
Per your instructions, **none of the packages contain**: `print` statements,
data loaders / `Dataset` construction, file reading/extraction code, or
plotting code. Everything here is designed to be called from your own
data-loading + logging + plotting layer.

```
codebase/
├── pinn/                          # Multi-Task Physics-Informed Neural Network
│   ├── config.py                  # physical constants, drift-flux closure, hyperparameters
│   ├── features.py                # FFT + statistical feature extraction, resampling
│   ├── splits.py                  # filename parsing + GroupKFold split logic (no file reads)
│   ├── models/
│   │   ├── tcn.py                 # Chomp1d, TemporalBlock, TemporalConvNet
│   │   └── pinn_model.py          # MultiTaskPINN (TCN + feature branch + PINN heads)
│   ├── losses/
│   │   ├── uncertainty_weighting.py   # homoscedastic uncertainty weighting (Kendall et al.)
│   │   ├── physics_loss.py            # drift-flux Darcy-Weisbach residual
│   │   ├── mass_conservation_loss.py  # steady incompressible continuity residual
│   │   └── total_loss.py              # combines all 5 task losses
│   └── training/
│       ├── trainer.py             # train_model() — one fold's epoch loop
│       ├── kfold_runner.py        # FoldBundle + run_kfold_train_val_test() — CV orchestration
│       └── evaluation.py          # evaluate_on_loader(), regression_metrics()
│
└── ablation/                      # Classical-ML baseline ablation study
    ├── config.py
    ├── models.py                  # get_classifiers(), get_regressors()
    ├── splits.py                  # make_split() — StratifiedKFold + held-out test
    ├── stats.py                   # confidence_interval(), clipped_asymmetric_error()
    └── training.py                # run_classification_ablation(), run_regression_ablation()
```

## What's intentionally NOT here

- **Data loading**: reading `.xlsx` files, building `MultiphaseFlowDataset`,
  constructing `DataLoader`s. `pinn/training/trainer.py` and
  `pinn/training/kfold_runner.py` take already-built `DataLoader`s /
  `FoldBundle`s as parameters — wire up your own data layer and pass them in.
- **Prints/logging**: none of these functions print. Hang your own logger
  off the returned `history` dicts / result dicts.
- **Plotting**: no matplotlib/seaborn. Everything returns plain
  dicts/arrays (`history`, `eval_results`, `summary_rows`) for you to plot
  however you like.

## Wiring it together (sketch)

```python
from pathlib import Path
from pinn.training import FoldBundle, run_kfold_train_val_test, evaluate_on_loader

# 1. Your own data layer builds one FoldBundle per fold:
#    train_loader, val_loader, fitted scaler_vsg/scaler_vsl,
#    dpdx_mean/std and q_mean/std (computed ONLY from that fold's train split).
fold_bundles = [ ... ]  # list[FoldBundle]

# 2. Train + cross-validate:
result = run_kfold_train_val_test(fold_bundles, models_dir=Path("models"))

# 3. Evaluate the best fold's model on your held-out test loader:
eval_results = evaluate_on_loader(
    best_model, test_loader,
    result["best_scalers"]["vsg"], result["best_scalers"]["vsl"],
    result["best_scalers"]["dpdx_mean"], result["best_scalers"]["dpdx_std"],
    result["best_scalers"]["q_mean"], result["best_scalers"]["q_std"],
)
```

## Notebooks not modularized here

- **`topk-video-retrieval.ipynb`** was left out. It's an inference/demo
  tool (loads video files, uses `ipywidgets`/`IPython.display` for an
  interactive UI) — it has no training loop, no loss computation, and no
  KFold logic, so it didn't match any item on the "keep" list, and its
  content is almost entirely file reading + UI + duplicate model
  definition (already covered by `pinn/models/`).
