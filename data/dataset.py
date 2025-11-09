"""
PyTorch Dataset class for flow regime data
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


class FlowRegimeDataset(Dataset):
    """Dataset for multi-task flow regime classification and velocity regression"""
    
    def __init__(self, data, scaler_features=None, scaler_pressure=None, 
                 scaler_vsg=None, scaler_vsl=None):
        self.data = data
        
        # Extract arrays
        pressure_windows = np.array([d['pressure_window'] for d in data])
        features = np.array([d['features'] for d in data])
        vsg = np.array([d['vsg'] for d in data]).reshape(-1, 1)
        vsl = np.array([d['vsl'] for d in data]).reshape(-1, 1)
        
        # Scale pressure windows
        if scaler_pressure is None:
            self.scaler_pressure = StandardScaler()
            pressure_windows = self.scaler_pressure.fit_transform(pressure_windows)
        else:
            self.scaler_pressure = scaler_pressure
            pressure_windows = self.scaler_pressure.transform(pressure_windows)
        
        # Scale features
        if scaler_features is None:
            self.scaler_features = StandardScaler()
            features = self.scaler_features.fit_transform(features)
        else:
            self.scaler_features = scaler_features
            features = self.scaler_features.transform(features)
        
        # Scale velocities separately using RobustScaler
        if scaler_vsg is None:
            self.scaler_vsg = RobustScaler()
            vsg_scaled = self.scaler_vsg.fit_transform(vsg)
        else:
            self.scaler_vsg = scaler_vsg
            vsg_scaled = self.scaler_vsg.transform(vsg)
        
        if scaler_vsl is None:
            self.scaler_vsl = RobustScaler()
            vsl_scaled = self.scaler_vsl.fit_transform(vsl)
        else:
            self.scaler_vsl = scaler_vsl
            vsl_scaled = self.scaler_vsl.transform(vsl)
        
        # Combine scaled velocities
        velocities_scaled = np.hstack([vsg_scaled, vsl_scaled])
        
        # Convert to tensors
        self.pressure_windows = torch.FloatTensor(pressure_windows)
        self.features = torch.FloatTensor(features)
        self.velocities_input = torch.FloatTensor(velocities_scaled)
        self.labels = torch.LongTensor([d['label'] for d in data])
        self.vsg_target = torch.FloatTensor(vsg_scaled)
        self.vsl_target = torch.FloatTensor(vsl_scaled)
        
        # Store raw velocities
        self.vsg_raw = vsg
        self.vsl_raw = vsl
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'pressure_window': self.pressure_windows[idx],
            'features': self.features[idx],
            'velocities': self.velocities_input[idx],
            'label': self.labels[idx],
            'vsg': self.vsg_target[idx],
            'vsl': self.vsl_target[idx]
        }