"""
Training function for multi-task learning
"""
import torch
import torch.nn as nn
from config import *


def train_fold_multitask(model, train_loader, val_loader, criterion_class, 
                         criterion_velocity, optimizer, scheduler, fold):
    """Training function for multi-task learning"""
    train_losses = []
    train_class_losses = []
    train_velocity_losses = []
    train_physics_losses = []
    train_accs = []
    
    val_losses = []
    val_class_losses = []
    val_velocity_losses = []
    val_accs = []
    val_velocity_maes = []
    
    best_val_acc = 0.0
    best_val_velocity_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_loss_class = 0.0
        train_loss_velocity = 0.0
        train_loss_physics = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if batch['label'].size(0) <= 1:
                continue
            
            pressure = batch['pressure_window'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            velocities = batch['velocities'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            vsg_target = batch['vsg'].to(DEVICE)
            vsl_target = batch['vsl'].to(DEVICE)
            
            velocity_target = torch.cat([vsg_target, vsl_target], dim=1)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_output, velocity_output, physics_output = model(pressure, features, velocities)
            
            # Multi-task losses
            loss_class = criterion_class(class_output, labels)
            loss_velocity = criterion_velocity(velocity_output, velocity_target)
            
            from models.losses import physics_loss
            loss_physics = physics_loss(pressure, vsg_target, vsl_target, physics_output,
                                       RHO_WATER, RHO_AIR_MEAN, GRAVITY)
            mean_pressure = pressure.mean(dim=1, keepdim=True)
            loss_data = nn.MSELoss()(physics_output, mean_pressure)
            
            # Combined loss
            loss = (LAMBDA_CLASS * loss_class + 
                   LAMBDA_VELOCITY * loss_velocity + 
                   LAMBDA_PHYSICS * loss_physics + 
                   LAMBDA_DATA * loss_data)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_class += loss_class.item()
            train_loss_velocity += loss_velocity.item()
            train_loss_physics += loss_physics.item()
            
            _, predicted = torch.max(class_output, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_class = train_loss_class / len(train_loader)
        avg_train_loss_velocity = train_loss_velocity / len(train_loader)
        avg_train_loss_physics = train_loss_physics / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_class = 0.0
        val_loss_velocity = 0.0
        val_correct = 0
        val_total = 0
        val_velocity_mae = 0.0
        val_velocity_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pressure = batch['pressure_window'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                velocities = batch['velocities'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                vsg_target = batch['vsg'].to(DEVICE)
                vsl_target = batch['vsl'].to(DEVICE)
                
                velocity_target = torch.cat([vsg_target, vsl_target], dim=1)
                
                class_output, velocity_output, physics_output = model(pressure, features, velocities)
                
                loss_class = criterion_class(class_output, labels)
                loss_velocity = criterion_velocity(velocity_output, velocity_target)
                
                from models.losses import physics_loss
                loss_physics = physics_loss(pressure, vsg_target, vsl_target, physics_output,
                                           RHO_WATER, RHO_AIR_MEAN, GRAVITY)
                mean_pressure = pressure.mean(dim=1, keepdim=True)
                loss_data = nn.MSELoss()(physics_output, mean_pressure)
                
                loss = (LAMBDA_CLASS * loss_class + 
                       LAMBDA_VELOCITY * loss_velocity + 
                       LAMBDA_PHYSICS * loss_physics + 
                       LAMBDA_DATA * loss_data)
                
                val_loss += loss.item()
                val_loss_class += loss_class.item()
                val_loss_velocity += loss_velocity.item()
                
                _, predicted = torch.max(class_output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                velocity_mae = torch.mean(torch.abs(velocity_output - velocity_target))
                val_velocity_mae += velocity_mae.item() * labels.size(0)
                val_velocity_count += labels.size(0)
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_class = val_loss_class / len(val_loader)
        avg_val_loss_velocity = val_loss_velocity / len(val_loader)
        avg_val_velocity_mae = val_velocity_mae / val_velocity_count if val_velocity_count > 0 else 0
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_class_losses.append(avg_train_loss_class)
        train_velocity_losses.append(avg_train_loss_velocity)
        train_physics_losses.append(avg_train_loss_physics)
        train_accs.append(train_acc)
        
        val_losses.append(avg_val_loss)
        val_class_losses.append(avg_val_loss_class)
        val_velocity_losses.append(avg_val_loss_velocity)
        val_accs.append(val_acc)
        val_velocity_maes.append(avg_val_velocity_mae)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Fold {fold}, Epoch [{epoch+1}/{EPOCHS}]')
            print(f'  Train - Loss: {avg_train_loss:.4f}, Class: {avg_train_loss_class:.4f}, '
                  f'Velocity: {avg_train_loss_velocity:.4f}, Physics: {avg_train_loss_physics:.4f}, '
                  f'Acc: {train_acc:.2f}%')
            print(f'  Val   - Loss: {avg_val_loss:.4f}, Class: {avg_val_loss_class:.4f}, '
                  f'Velocity: {avg_val_loss_velocity:.4f}, Acc: {val_acc:.2f}%, '
                  f'Vel MAE: {avg_val_velocity_mae:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_velocity_mae = avg_val_velocity_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_velocity_mae': avg_val_velocity_mae
            }, f'best_multitask_pinn_fold_{fold}.pth')
        else:
            patience_counter += 1
    
    return (train_losses, train_class_losses, train_velocity_losses, train_physics_losses, train_accs,
            val_losses, val_class_losses, val_velocity_losses, val_accs, val_velocity_maes,
            best_val_acc, best_val_velocity_mae)