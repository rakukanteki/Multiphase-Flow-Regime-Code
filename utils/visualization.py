"""
Visualization utilities for training results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from config import K_FOLDS


def visualize_multitask_results(all_fold_results, all_predictions, all_true_labels,
                                all_velocity_predictions, all_velocity_targets, class_names):
    """
    Comprehensive visualization for multi-task learning results
    
    Args:
        all_fold_results: List of fold training results
        all_predictions: List of all predictions
        all_true_labels: List of all true labels
        all_velocity_predictions: Array of velocity predictions
        all_velocity_targets: Array of velocity targets
        class_names: List of class names
    """
    n_folds = len(all_fold_results)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for fold_idx, fold_result in enumerate(all_fold_results):
        train_losses = fold_result[0]
        val_losses = fold_result[5]
        ax1.plot(train_losses, alpha=0.3, color='blue', linewidth=1)
        ax1.plot(val_losses, alpha=0.3, color='red', linewidth=1)
    
    # Plot average lines
    avg_train = np.mean([fold[0] for fold in all_fold_results], axis=0)
    avg_val = np.mean([fold[5] for fold in all_fold_results], axis=0)
    ax1.plot(avg_train, color='blue', linewidth=2, label='Avg Train')
    ax1.plot(avg_val, color='red', linewidth=2, label='Avg Val')
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training and Validation Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    for fold_idx, fold_result in enumerate(all_fold_results):
        train_accs = fold_result[4]
        val_accs = fold_result[8]
        ax2.plot(train_accs, alpha=0.3, color='blue', linewidth=1)
        ax2.plot(val_accs, alpha=0.3, color='red', linewidth=1)
    
    # Plot average lines
    avg_train_acc = np.mean([fold[4] for fold in all_fold_results], axis=0)
    avg_val_acc = np.mean([fold[8] for fold in all_fold_results], axis=0)
    ax2.plot(avg_train_acc, color='blue', linewidth=2, label='Avg Train')
    ax2.plot(avg_val_acc, color='red', linewidth=2, label='Avg Val')
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity MAE
    ax3 = fig.add_subplot(gs[0, 2])
    for fold_idx, fold_result in enumerate(all_fold_results):
        val_velocity_maes = fold_result[9]
        ax3.plot(val_velocity_maes, alpha=0.5, label=f'Fold {fold_idx+1}', linewidth=1)
    
    avg_vel_mae = np.mean([fold[9] for fold in all_fold_results], axis=0)
    ax3.plot(avg_vel_mae, color='black', linewidth=2, label='Average', linestyle='--')
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Velocity MAE', fontweight='bold')
    ax3.set_title('Validation Velocity MAE', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(all_true_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontweight='bold')
    ax4.set_ylabel('True', fontweight='bold')
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 5. Velocity Predictions - Vsg
    ax5 = fig.add_subplot(gs[1, 1])
    all_velocity_predictions = np.array(all_velocity_predictions)
    all_velocity_targets = np.array(all_velocity_targets)
    
    ax5.scatter(all_velocity_targets[:, 0], all_velocity_predictions[:, 0], 
                alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(all_velocity_targets[:, 0].min(), all_velocity_predictions[:, 0].min())
    max_val = max(all_velocity_targets[:, 0].max(), all_velocity_predictions[:, 0].max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    vsg_mae = mean_absolute_error(all_velocity_targets[:, 0], all_velocity_predictions[:, 0])
    ax5.set_xlabel('True Vsg (m/s)', fontweight='bold')
    ax5.set_ylabel('Predicted Vsg (m/s)', fontweight='bold')
    ax5.set_title(f'Vsg Predictions (MAE: {vsg_mae:.4f} m/s)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Velocity Predictions - Vsl
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(all_velocity_targets[:, 1], all_velocity_predictions[:, 1], 
                alpha=0.5, s=20, edgecolors='black', linewidths=0.5, color='green')
    
    min_val = min(all_velocity_targets[:, 1].min(), all_velocity_predictions[:, 1].min())
    max_val = max(all_velocity_targets[:, 1].max(), all_velocity_predictions[:, 1].max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    vsl_mae = mean_absolute_error(all_velocity_targets[:, 1], all_velocity_predictions[:, 1])
    ax6.set_xlabel('True Vsl (m/s)', fontweight='bold')
    ax6.set_ylabel('Predicted Vsl (m/s)', fontweight='bold')
    ax6.set_title(f'Vsl Predictions (MAE: {vsl_mae:.4f} m/s)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Per-class precision
    ax7 = fig.add_subplot(gs[2, 0])
    report = classification_report(all_true_labels, all_predictions, 
                                   target_names=class_names, output_dict=True)
    class_precisions = [report[name]['precision'] for name in class_names]
    bars = ax7.bar(class_names, class_precisions, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   edgecolor='black', linewidth=1.5)
    
    for bar, prec in zip(bars, class_precisions):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{prec:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax7.set_ylabel('Precision', fontweight='bold')
    ax7.set_title('Per-Class Precision', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 1.1])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Fold-wise best accuracies
    ax8 = fig.add_subplot(gs[2, 1])
    fold_accs = [fold_result[10] for fold_result in all_fold_results]
    bars = ax8.bar(range(1, n_folds+1), fold_accs, color='#9b59b6', 
                   edgecolor='black', linewidth=1.5)
    
    mean_acc = np.mean(fold_accs)
    ax8.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_acc:.2f}%')
    
    for bar, acc in zip(bars, fold_accs):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax8.set_xlabel('Fold', fontweight='bold')
    ax8.set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    ax8.set_title('Fold-wise Best Accuracies', fontsize=12, fontweight='bold')
    ax8.set_xticks(range(1, n_folds+1))
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Overall metrics text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    overall_acc = 100 * np.mean(np.array(all_predictions) == np.array(all_true_labels))
    vsg_mae = mean_absolute_error(all_velocity_targets[:, 0], all_velocity_predictions[:, 0])
    vsl_mae = mean_absolute_error(all_velocity_targets[:, 1], all_velocity_predictions[:, 1])
    vsg_rmse = np.sqrt(np.mean((all_velocity_targets[:, 0] - all_velocity_predictions[:, 0])**2))
    vsl_rmse = np.sqrt(np.mean((all_velocity_targets[:, 1] - all_velocity_predictions[:, 1])**2))
    
    metrics_text = "═══════════════════════════════\n"
    metrics_text += "      OVERALL METRICS\n"
    metrics_text += "═══════════════════════════════\n\n"
    metrics_text += f"Classification Accuracy: {overall_acc:.2f}%\n\n"
    metrics_text += "Velocity Regression Metrics:\n"
    metrics_text += f"  Vsg MAE:  {vsg_mae:.4f} m/s\n"
    metrics_text += f"  Vsg RMSE: {vsg_rmse:.4f} m/s\n"
    metrics_text += f"  Vsl MAE:  {vsl_mae:.4f} m/s\n"
    metrics_text += f"  Vsl RMSE: {vsl_rmse:.4f} m/s\n\n"
    metrics_text += "───────────────────────────────\n"
    metrics_text += "Per-Class Performance:\n"
    metrics_text += "───────────────────────────────\n"
    
    for name in class_names:
        precision = report[name]['precision']
        recall = report[name]['recall']
        f1 = report[name]['f1-score']
        support = int(report[name]['support'])
        metrics_text += f"\n{name}:\n"
        metrics_text += f"  Precision: {precision:.3f}\n"
        metrics_text += f"  Recall:    {recall:.3f}\n"
        metrics_text += f"  F1-Score:  {f1:.3f}\n"
        metrics_text += f"  Support:   {support}\n"
    
    ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3,
                      edgecolor='black', linewidth=2))
    
    plt.savefig('multitask_training_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved as 'multitask_training_results.png'")
    plt.show()
    
    # Print detailed console output
    print("\n" + "="*70)
    print("MULTI-TASK K-FOLD CROSS-VALIDATION RESULTS")
    print("="*70)
    
    print("\nOverall Classification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                               target_names=class_names, digits=4))
    
    print("\nFold-wise Results:")
    print("-" * 70)
    for fold_idx, fold_result in enumerate(all_fold_results):
        best_acc = fold_result[10]
        best_vel_mae = fold_result[11]
        print(f"Fold {fold_idx + 1}:")
        print(f"  Best Validation Accuracy:     {best_acc:.2f}%")
        print(f"  Best Velocity MAE (scaled):   {best_vel_mae:.4f}")
    
    print("-" * 70)
    print(f"Mean Accuracy:       {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")
    print(f"Mean Velocity MAE:   {np.mean([f[11] for f in all_fold_results]):.4f} ± "
          f"{np.std([f[11] for f in all_fold_results]):.4f}")
    
    print("\n" + "="*70)