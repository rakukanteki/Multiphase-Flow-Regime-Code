"""
Data preprocessing and feature extraction
"""
import numpy as np
from scipy.fft import fft, fftfreq


def extract_features(pressure_window):
    """Extract statistical and frequency domain features from pressure window"""
    features = []
    
    # Statistical features
    features.append(np.mean(pressure_window))
    features.append(np.std(pressure_window))
    features.append(np.max(pressure_window) - np.min(pressure_window))
    
    # Gradient features
    gradient = np.gradient(pressure_window)
    features.append(np.mean(gradient))
    features.append(np.std(gradient))
    features.append(np.max(np.abs(gradient)))
    
    # Frequency domain features
    if len(pressure_window) > 4:
        freqs = fftfreq(len(pressure_window), d=0.5)
        fft_vals = np.abs(fft(pressure_window))
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]
        
        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            features.append(positive_freqs[dominant_freq_idx])
            features.append(positive_fft[dominant_freq_idx])
        else:
            features.append(0.0)
            features.append(0.0)
    else:
        features.append(0.0)
        features.append(0.0)
    
    return np.array(features)


def load_data(base_dir, sub_folders, window_size, stride):
    """Load and preprocess all data from folders"""
    import os
    import pandas as pd
    from utils.helpers import extract_velocities_from_filename
    
    all_data = []
    
    for idx, folder in enumerate(sub_folders):
        folder_path = os.path.join(base_dir, folder)
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            
            if 'Pressure (barA)' in df.columns:
                pressure = df['Pressure (barA)'].values
                vsg, vsl = extract_velocities_from_filename(file)
                
                # Create sliding windows
                for i in range(0, len(pressure) - window_size + 1, stride):
                    window = pressure[i:i+window_size]
                    
                    if len(window) == window_size:
                        features = extract_features(window)
                        all_data.append({
                            'pressure_window': window,
                            'features': features,
                            'vsg': vsg,
                            'vsl': vsl,
                            'label': idx
                        })
    
    return all_data