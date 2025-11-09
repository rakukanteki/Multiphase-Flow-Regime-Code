"""
K-Fold cross-validation training
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config import *
from data.dataset import FlowRegimeDataset
from models.pinn import MultiTaskPINN
from training.trainer import train_fold_multitask
from utils.visualization import visualize_multitask_results


def kfold_train_multitask(data):
    """K-Fold cross-validation for multi-task learning"""
    print("Starting K-Fold Cross-Validation...")
    print(f"Total samples: {len(data)}")
    
    labels = np.array([d['label'] for d in data])
    indices = np.arange(len(data))
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    all_fold_results = []
    all_predictions = []
    all_true_labels = []
    all_velocity_predictions = []
    all_velocity_targets = []
    
    best_model = None
    best_val_acc_overall = 0.0
    best_fold = None
    best_scalers = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'='*60}")
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        train_dataset = FlowRegimeDataset(train_data)
        val_dataset = FlowRegimeDataset(val_data, 
                                        scaler_features=train_dataset.scaler_features,
                                        scaler_pressure=train_dataset.scaler_pressure,
                                        scaler_vsg=train_dataset.scaler_vsg,
                                        scaler_vsl=train_dataset.scaler_vsl)
        
        # Reproducible DataLoader
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                 drop_last=True, generator=g, 
                                 worker_init_fn=lambda _: np.random.seed(RANDOM_SEED))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = MultiTaskPINN(input_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE, 
                             num_classes=NUM_CLASSES).to(DEVICE)
        
        # Loss functions
        class_counts = np.bincount(labels[train_idx])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        criterion_class = nn.CrossEntropyLoss(weight=class_weights)
        criterion_velocity = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                          patience=10, verbose=True)
        
        fold_results = train_fold_multitask(
            model, train_loader, val_loader, criterion_class, criterion_velocity,
            optimizer, scheduler, fold + 1
        )
        
        all_fold_results.append(fold_results)
        
        # Load best model for evaluation
        checkpoint = torch.load(f'best_multitask_pinn_fold_{fold + 1}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate on validation set
        fold_preds = []
        fold_labels = []
        fold_vel_preds_scaled = []
        fold_vel_targets_scaled = []
        
        with torch.no_grad():
            for batch in val_loader:
                pressure = batch['pressure_window'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                velocities = batch['velocities'].to(DEVICE)
                labels_batch = batch['label'].to(DEVICE)
                vsg_target = batch['vsg'].to(DEVICE)
                vsl_target = batch['vsl'].to(DEVICE)
                
                velocity_target = torch.cat([vsg_target, vsl_target], dim=1)
                
                class_output, velocity_output, _ = model(pressure, features, velocities)
                _, predicted = torch.max(class_output, 1)
                
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels_batch.cpu().numpy())
                fold_vel_preds_scaled.extend(velocity_output.cpu().numpy())
                fold_vel_targets_scaled.extend(velocity_target.cpu().numpy())
        
        # Inverse transform predictions to original scale
        fold_vel_preds_scaled = np.array(fold_vel_preds_scaled)
        fold_vel_targets_scaled = np.array(fold_vel_targets_scaled)
        
        vsg_pred_original = val_dataset.scaler_vsg.inverse_transform(
            fold_vel_preds_scaled[:, 0].reshape(-1, 1)
        )
        vsg_target_original = val_dataset.scaler_vsg.inverse_transform(
            fold_vel_targets_scaled[:, 0].reshape(-1, 1)
        )
        
        vsl_pred_original = val_dataset.scaler_vsl.inverse_transform(
            fold_vel_preds_scaled[:, 1].reshape(-1, 1)
        )
        vsl_target_original = val_dataset.scaler_vsl.inverse_transform(
            fold_vel_targets_scaled[:, 1].reshape(-1, 1)
        )
        
        fold_vel_preds = np.hstack([vsg_pred_original, vsl_pred_original])
        fold_vel_targets = np.hstack([vsg_target_original, vsl_target_original])
        
        all_predictions.extend(fold_preds)
        all_true_labels.extend(fold_labels)
        all_velocity_predictions.extend(fold_vel_preds)
        all_velocity_targets.extend(fold_vel_targets)
        
        best_val_acc = fold_results[10]
        best_val_velocity_mae = fold_results[11]
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"  Best Velocity MAE (scaled): {best_val_velocity_mae:.4f}")
        
        if best_val_acc > best_val_acc_overall:
            best_val_acc_overall = best_val_acc
            best_model = model
            best_fold = fold + 1
            best_scalers = {
                'vsg': val_dataset.scaler_vsg,
                'vsl': val_dataset.scaler_vsl,
                'features': val_dataset.scaler_features,
                'pressure': val_dataset.scaler_pressure
            }
    
    print(f"\n✅ Best model is from Fold {best_fold} with Accuracy: {best_val_acc_overall:.2f}%")
    
    # Visualization
    visualize_multitask_results(all_fold_results, all_predictions, all_true_labels, 
                                all_velocity_predictions, all_velocity_targets, CLASS_NAMES)
    
    return best_model, all_fold_results, best_scalers