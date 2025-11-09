"""
Training and loss functions module
"""
from .trainer import train_fold_multitask, kfold_train_multitask
from .losses import physics_loss, combined_loss

__all__ = ['train_fold_multitask', 'kfold_train_multitask', 'physics_loss', 'combined_loss']