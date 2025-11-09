"""
Configuration file for Multi-Task PINN Flow Regime Classification
"""
import torch
import os

# ==================== PATHS ====================
BASE_DIR = r"D:\Research\Multiphase Visual Twin\Papers\Paper1\Project\20Hz-Training-Dataset"
SUB_FOLDERS = ["Dispersed-Flow", "Plug-Flow", "Slug-Flow"]
VIDEO_LIBRARY_CSV = r"D:\Research\Multiphase Visual Twin\Papers\Paper1\Project\20Hz-Training-Dataset\Code\video_library.csv"
TEST_DATA_DIR = r"D:\Research\Multiphase Visual Twin\Papers\Paper1\Project\20Hz-Training-Dataset\Code\External-Validation\Good"

# ==================== DEVICE ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== PHYSICS CONSTANTS ====================
PIPE_DIAMETER = 2 * 0.0254  # meters
PIPE_LENGTH = 7.25  # meters
GRAVITY = 9.81  # m/s^2
RHO_WATER = 998  # kg/m^3
RHO_AIR_MEAN = 1.49  # kg/m^3

# ==================== MODEL HYPERPARAMETERS ====================
WINDOW_SIZE = 40
STRIDE = 20
HIDDEN_SIZE = 128
NUM_CLASSES = 3

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

# ==================== LOSS WEIGHTS ====================
LAMBDA_CLASS = 0.7
LAMBDA_PHYSICS = 0.3
LAMBDA_DATA = 0.2
LAMBDA_VELOCITY = 0.7

# ==================== CROSS-VALIDATION ====================
K_FOLDS = 5
RANDOM_SEED = 42

# ==================== RETRIEVAL PARAMETERS ====================
TOP_K = 3  # Number of similar videos to retrieve

# ==================== MODEL PATHS ====================
MODEL_PATH = "best_multitask_pinn_fold_1.pth"
SCALERS_PATH = "best_model_scalers.pkl"

# ==================== CLASS NAMES ====================
CLASS_NAMES = ["Dispersed Flow", "Plug Flow", "Slug Flow"]