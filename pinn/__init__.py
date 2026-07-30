"""
Multi-Task Physics-Informed Neural Network (PINN) package.

Predicts flow regime classification + superficial gas/liquid velocity
(Vsg, Vsl) regression from a pressure time-series, constrained by a
drift-flux Darcy-Weisbach physics residual and a mass-conservation
residual, combined via learned homoscedastic uncertainty weighting.

This package intentionally does NOT include:
    - data loading / file I/O (bring your own Dataset/DataLoader)
    - print/logging statements (wire up your own logger if desired)
    - plotting code

Sub-packages:
    config      -- physical constants & hyperparameters
    features    -- FFT / statistical feature extraction, resampling
    splits      -- GroupKFold split logic over filenames (no file reads)
    models      -- TCN backbone + MultiTaskPINN
    losses      -- uncertainty weighting, physics loss, mass-conservation
                   loss, combined total loss
    training    -- epoch training loop, kfold orchestration, evaluation
"""
