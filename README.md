# Multi-Task Physics-Informed Neural Network for Flow Regime Classification

A deep learning system for classifying multiphase flow regimes and regressing superficial velocities using pressure signals, with video retrieval capabilities based on learned embeddings.

## 🌟 Features

- **Multi-Task Learning**: Simultaneous flow regime classification, velocity regression, and physics-informed pressure prediction
- **Physics-Informed Neural Network (PINN)**: Incorporates physical laws into the loss function for improved generalization
- **K-Fold Cross-Validation**: Robust model evaluation with 5-fold stratified cross-validation
- **Video Retrieval System**: Content-based video retrieval using learned embeddings and similarity metrics
- **Comprehensive Visualization**: Training curves, confusion matrices, velocity predictions, error distributions, and video frame extraction
- **Reproducible Results**: Fixed random seeds and deterministic training

## 📁 Project Structure
```
flow-regime-classification/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Main training script
├── inference.py              # Main inference/retrieval script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset class
│   └── preprocessing.py     # Feature extraction and data loading
│
├── models/
│   ├── __init__.py
│   ├── pinn.py              # Multi-Task PINN architecture
│   └── losses.py            # Custom loss functions
│
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Training loop for single fold
│   └── kfold.py             # K-fold cross-validation logic
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Utility functions (seed, similarity)
│   └── visualization.py     # Plotting and visualization
│
└── inference/
    ├── __init__.py
    └── retrieval.py         # Video retrieval system
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/flow-regime-classification.git
cd flow-regime-classification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Update paths in `config.py`:**
   - `BASE_DIR`: Path to your training data folder
   - `VIDEO_LIBRARY_CSV`: Path to video library CSV file
   - `TEST_DATA_DIR`: Path to test/validation data

### Training

Run the training script with K-fold cross-validation:
```bash
python train.py
```

**Training Outputs:**
- `best_multitask_pinn_fold_*.pth`: Best model checkpoint from each fold
- `best_model_scalers.pkl`: Fitted scalers for preprocessing (required for inference)
- `multitask_training_results.png`: Comprehensive visualization of training metrics

**Training takes approximately:** 2-4 hours on GPU (depends on dataset size)

### Inference & Video Retrieval

Run the inference script to retrieve similar videos:
```bash
python inference.py
```

**Inference Outputs:**
- `retrieval_results_*.png`: Visualization for each test file
- `video_frame_grid.png`: Extracted frames from retrieved videos
- Console output with predictions and similarity scores

## 📊 Model Architecture

The **Multi-Task PINN** consists of:

### Shared Backbone
1. **CNN Layers**: Process pressure signal windows
   - Conv1D (1→32 filters) + BatchNorm + MaxPool
   - Conv1D (32→64 filters) + BatchNorm + Dropout
2. **Feature Processing Branch**: Statistical and frequency features (8 features)
3. **Velocity Processing Branch**: Superficial velocities (Vsg, Vsl)
4. **Shared Representation Layer**: Combines all features (128 hidden units)

### Task-Specific Heads
1. **Classification Head**: 3-class flow regime classifier
2. **Velocity Regression Head**: Predicts Vsg and Vsl
3. **Physics-Based Head**: Predicts pressure gradient using physical laws

### Loss Function
```
Total Loss = λ_class * L_class + λ_velocity * L_velocity + λ_physics * L_physics + λ_data * L_data
```

Where:
- `L_class`: Cross-entropy loss (weighted by class frequency)
- `L_velocity`: MSE loss for velocity predictions
- `L_physics`: Physics-informed loss (pressure gradient residual)
- `L_data`: Data consistency loss (pressure reconstruction)

## ⚙️ Configuration

Key hyperparameters in `config.py`:
```python
# Signal processing
WINDOW_SIZE = 40        # Pressure signal window size
STRIDE = 20             # Sliding window stride

# Training
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0005
K_FOLDS = 5

# Loss weights
LAMBDA_CLASS = 0.7      # Classification loss weight
LAMBDA_VELOCITY = 0.7   # Velocity regression loss weight
LAMBDA_PHYSICS = 0.3    # Physics loss weight
LAMBDA_DATA = 0.2       # Data consistency loss weight

# Retrieval
TOP_K = 5              # Number of similar videos to retrieve
```

## 📈 Expected Performance

Typical results on our dataset:

| Metric | Value |
|--------|-------|
| Classification Accuracy | ~95% |
| Vsg MAE | ~0.02 m/s |
| Vsl MAE | ~0.03 m/s |
| Training Time (per fold) | ~30 min (GPU) |

## 📁 Data Format

### Training Data Structure
```
BASE_DIR/
├── Dispersed-Flow/
│   ├── DispersedFlow-Vsg=X.XX-Vsl=Y.YY.xlsx
│   └── ...
├── Plug-Flow/
│   ├── PlugFlow-Vsg=X.XX-Vsl=Y.YY.xlsx
│   └── ...
└── Slug-Flow/
    ├── SlugFlow-Vsg=X.XX-Vsl=Y.YY.xlsx
    └── ...
```

**Excel File Requirements:**
- Must contain column: `Pressure (barA)` or `Pressure/bar`
- Filename must include velocities: `FlowType-Vsg=X.XX-Vsl=Y.YY.xlsx`

### Video Library CSV

Required columns:
- `filename`: Video filename
- `video_path`: Full path to video file
- `flow_regime`: Flow regime name (string)
- `flow_regime_idx`: Flow regime index (0, 1, or 2)
- `vsg`: Superficial gas velocity (m/s)
- `vsl`: Superficial liquid velocity (m/s)

**Example:**
```csv
filename,video_path,flow_regime,flow_regime_idx,vsg,vsl
slug_flow_1.mp4,/path/to/slug_flow_1.mp4,Slug Flow,2,0.29,1.1
dispersed_flow_1.mp4,/path/to/dispersed_flow_1.mp4,Dispersed Flow,0,0.05,2.88
```

## 🔬 Physics-Informed Component

The model incorporates multiphase flow physics through:

**Void Fraction (α):**
```
α = Vsg / (Vsg + Vsl)
```

**Two-Phase Density (ρ_tp):**
```
ρ_tp = α * ρ_gas + (1 - α) * ρ_liquid
```

**Gravitational Pressure Gradient:**
```
dP/dz = ρ_tp * g / 100000
```

This physics-informed loss guides the model to learn physically consistent representations.

## 🎯 Flow Regimes

The system classifies three flow regimes:

1. **Dispersed Flow** (Class 0): Gas bubbles dispersed in continuous liquid
2. **Plug Flow** (Class 1): Alternating plugs of gas and liquid
3. **Slug Flow** (Class 2): Large gas pockets (Taylor bubbles) with liquid slugs

## 🔍 Video Retrieval

The retrieval system uses:
- **Embedding Space**: 128-dimensional learned representations from the shared layer
- **Similarity Metrics**: Cosine similarity (default) or Euclidean distance
- **Filtering**: Retrieves videos only from the predicted flow regime class
- **Ranking**: Returns top-K most similar videos with similarity scores

## 🛠️ Troubleshooting

### Common Issues

1. **FileNotFoundError: Scalers not found**
```
   Solution: Run training first to generate scalers
```

2. **CUDA Out of Memory**
```python
   # Reduce batch size in config.py
   BATCH_SIZE = 8
```

3. **No pressure column found**
```
   Solution: Ensure Excel files have 'Pressure (barA)' or 'Pressure/bar' column
```

4. **Video playback issues**
```bash
   # Install video codecs
   pip install opencv-python imageio imageio-ffmpeg
```

## 🙏 Acknowledgments

- Physics equations based on standard multiphase flow literature
- CNN architecture inspired by time-series classification methods
- Multi-task learning framework adapted from computer vision applications

## 📞 Support

For questions or issues:
- Email: [radwankhondokar20@gmail.com](radwankhondokar20@gmail.com)
