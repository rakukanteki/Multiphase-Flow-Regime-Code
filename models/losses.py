"""
Loss functions for multi-task learning
"""
import torch
import torch.nn as nn


def physics_loss(pressure_window, vsg, vsl, physics_pred, 
                 rho_water=998, rho_air=1.49, gravity=9.81):
    """
    Physics-informed loss based on pressure gradient
    
    Args:
        pressure_window: Pressure signal window
        vsg: Superficial gas velocity
        vsl: Superficial liquid velocity
        physics_pred: Predicted pressure gradient
        rho_water: Water density (kg/m^3)
        rho_air: Air density (kg/m^3)
        gravity: Gravitational acceleration (m/s^2)
    
    Returns:
        Physics loss value
    """
    # Calculate pressure gradient
    pressure_grad = torch.gradient(pressure_window, dim=1)[0]
    mean_pressure_grad = pressure_grad.mean(dim=1, keepdim=True)
    
    # Calculate mixture properties
    vm = vsg + vsl
    void_fraction = vsg / (vm + 1e-8)
    rho_tp = void_fraction * rho_air + (1 - void_fraction) * rho_water
    
    # Theoretical pressure gradient (gravitational component)
    dp_dz_grav = rho_tp * gravity / 100000
    
    # Residual between prediction and theory
    residual = torch.abs(physics_pred - dp_dz_grav)
    
    return residual.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, lambda_class=0.7, lambda_velocity=0.7, 
                 lambda_physics=0.3, lambda_data=0.2):
        super(MultiTaskLoss, self).__init__()
        self.lambda_class = lambda_class
        self.lambda_velocity = lambda_velocity
        self.lambda_physics = lambda_physics
        self.lambda_data = lambda_data
        
        self.criterion_class = None  # Set externally with class weights
        self.criterion_velocity = nn.MSELoss()
        self.criterion_data = nn.MSELoss()
    
    def forward(self, class_output, velocity_output, physics_output, 
                labels, velocity_target, pressure, vsg, vsl):
        """Calculate combined loss"""
        # Classification loss
        loss_class = self.criterion_class(class_output, labels)
        
        # Velocity regression loss
        loss_velocity = self.criterion_velocity(velocity_output, velocity_target)
        
        # Physics loss
        loss_physics = physics_loss(pressure, vsg, vsl, physics_output)
        
        # Data consistency loss
        mean_pressure = pressure.mean(dim=1, keepdim=True)
        loss_data = self.criterion_data(physics_output, mean_pressure)
        
        # Combined loss
        total_loss = (self.lambda_class * loss_class + 
                     self.lambda_velocity * loss_velocity + 
                     self.lambda_physics * loss_physics + 
                     self.lambda_data * loss_data)
        
        return total_loss, loss_class, loss_velocity, loss_physics, loss_data