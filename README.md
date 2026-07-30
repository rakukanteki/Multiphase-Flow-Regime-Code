# Physics-Guided Multiphase Flow Metering and Regime Identification Using Multi-Task Learning with Condition-Aware Visualization

## Abstract:
Accurate recognition of gas-liquid two-phase flow regimes is critical for ensuring the safety, efficiency, and reliability of multiphase flow measurements. Traditional data-driven methods often overlook fundamental physical principles, limiting interpretability and generalization across varying operating conditions. This study proposes a Multi-Task Physics-Informed Neural Network (MTPINN) for simultaneous multiphase flow metering and flow regime classification by embedding momentum-based physical constraints into the learning process. The framework integrates CNN-based temporal pressure encoding, FFT-derived spectral features, and physics-informed regularization to produce physically consistent predictions. Evaluated under leakage-free five-fold GroupKFold cross-validation, the proposed model achieved an accuracy of 95.96% ± 3.13%, while attaining 92.86% classification accuracy and a weighted F1-score of 92.86%. The velocity regression module achieved mean absolute errors of 0.0537 m/s for gas superficial velocity (Vsg) and 0.2557 m/s for liquid superficial velocity (Vsl), enabling accurate non-intrusive flow metering. The predicted velocities are further exploited as operating-condition descriptors for condition-aware video retrieval, where the developed real-time system achieved 76.43% Top-1. The proposed physics-guided framework improves prediction accuracy, physical consistency, and interpretability while providing a practical foundation for intelligent multiphase flow monitoring and digital-twin applications.

## Contributions:
1.	A physics-constrained measurement learning framework is proposed, where governing momentum-balance relationships are embedded into a data-driven model to enforce physically consistent estimation of flow regimes and operating conditions from pressure signals.
2.	A unified multi-task measurement model is developed for simultaneous flow regime identification and superficial velocity estimation, enabling joint inference of discrete and continuous flow characteristics within a single physically guided framework.
3.	A condition-aware measurement validation mechanism is proposed, where predicted operating conditions are used to retrieve corresponding flow visualizations, enabling human-interpretable verification of measurement outputs.
4.	The framework enables real-time monitoring by linking sensor data, predicted operating conditions, and visual flow states to support digital-twin applications.

## Methodology Figure:
![Experimental Methodology](/assets/methodology.png)

## Model Architecture:
![Network Architecture](/assets/architectural.png)

## Performance Metrics:
##### LEARNED UNCERTAINTY-WEIGHTING SIGMAS (Best Fold, Final Epoch)
| Task | Learned Sigma (σ) |
|------|------------------:|
| Classification | **0.58767** |
| Gas Superficial Velocity (Vsg) | **0.69532** |
| Liquid Superficial Velocity (Vsl) | **0.58659** |
| Physics Constraint | **0.81358** |
| Mass Conservation Constraint | **0.64710** |

##### CLASSIFICATION REPORT (Test Split, Held-Out, File-Level, Best-Fold Model)
| Class | Precision | Recall | F1-Score | Support |
|--------|----------:|-------:|---------:|--------:|
| Dispersed Flow | 1.0000 | 1.0000 | 1.0000 | 18 |
| Plug Flow | 0.8947 | 0.8947 | 0.8947 | 19 |
| Slug Flow | 0.8947 | 0.8947 | 0.8947 | 19 |

| Metric | Precision | Recall | F1-Score | Support |
|--------|----------:|-------:|---------:|--------:|
| **Accuracy** | - | - | **0.9286** | 56 |
| **Macro Avg** | 0.9298 | 0.9298 | 0.9298 | 56 |
| **Weighted Avg** | 0.9286 | 0.9286 | 0.9286 | 56 |

## Top-K Visual Retrieval:


## Codebase Structure:
```
Multiphase-Flow-Regime-Code/
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
|    ├── config.py
|    ├── models.py                  # get_classifiers(), get_regressors()
|    ├── splits.py                  # make_split() — StratifiedKFold + held-out test
|    ├── stats.py                   # confidence_interval(), clipped_asymmetric_error()
|    └── training.py                # run_classification_ablation(), run_regression_ablation()
|
└── results/                       # Contains classical ML results, ablation study, Our model performance.
└── models/                        # Contains the trained models.
└── retrieval/                     # TopK visual retrieval files.
    ├── demo.py
    ├── display.py                  
    ├── player.py                  
    ├── system.py                  
    └── types.py               
```