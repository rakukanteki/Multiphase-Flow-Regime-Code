"""
Main training script
"""
import pickle
from config import *
from utils.helpers import set_seed
from data.preprocessing import load_data
from training.kfold import kfold_train_multitask


def main():
    """Main training pipeline"""
    print("="*70)
    print(" MULTI-TASK PINN TRAINING FOR FLOW REGIME CLASSIFICATION")
    print("="*70)
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    print(f"\n✓ Random seed set to {RANDOM_SEED}")
    print(f"✓ Device: {DEVICE}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    data = load_data(BASE_DIR, SUB_FOLDERS, WINDOW_SIZE, STRIDE)
    print(f"✓ Total samples: {len(data)}")
    
    # Train with K-fold cross-validation
    print("\n" + "="*70)
    print("STARTING K-FOLD CROSS-VALIDATION")
    print("="*70)
    best_model, fold_results, scalers = kfold_train_multitask(data)
    
    # Save scalers for inference
    scalers_path = 'best_model_scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n✅ Best model saved")
    print(f"✅ Scalers saved to '{scalers_path}'")
    print(f"✅ Visualizations saved:")
    print("   - multitask_training_results.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()