"""
Multi-Task Physics-Informed Neural Network model
"""
import torch
import torch.nn as nn


class MultiTaskPINN(nn.Module):
    """
    Multi-Task PINN for:
    1. Flow regime classification
    2. Velocity regression (Vsg, Vsl)
    3. Physics-based pressure prediction
    """
    
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super(MultiTaskPINN, self).__init__()
        
        # Shared CNN backbone for pressure signal processing
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4)
        
        # Feature processing
        self.fc_features = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Velocity processing
        self.fc_velocities = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Calculate combined feature size
        conv_output_size = 64 * (input_size // 2)
        combined_size = conv_output_size + 64 + 32
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Task 1: Flow Regime Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Task 2: Velocity Regression Head
        self.velocity_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: [Vsg, Vsl]
        )
        
        # Task 3: Physics-based pressure prediction
        self.physics_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, pressure_window, features, velocities):
        # Process pressure signal through CNN
        x = pressure_window.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        # Process features and velocities
        feat = self.fc_features(features)
        vel = self.fc_velocities(velocities)
        
        # Combine all features
        combined = torch.cat([x, feat, vel], dim=1)
        
        # Shared representation
        shared_repr = self.shared_layer(combined)
        
        # Multi-task outputs
        class_output = self.classifier(shared_repr)
        velocity_output = self.velocity_regressor(shared_repr)
        physics_output = self.physics_net(shared_repr)
        
        return class_output, velocity_output, physics_output