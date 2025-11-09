"""
Video retrieval system for flow regime classification
"""
import torch
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import *
from models.pinn import MultiTaskPINN
from data.preprocessing import extract_features
from utils.helpers import extract_velocities_from_filename, cosine_similarity, euclidean_distance


class VideoRetrievalSystem:
    """System for retrieving similar flow regime videos based on learned embeddings"""
    
    def __init__(self, model_path, scalers_path, video_library_csv, device=DEVICE):
        """
        Initialize the video retrieval system
        
        Args:
            model_path: Path to trained model checkpoint
            scalers_path: Path to saved scalers
            video_library_csv: Path to video library CSV
            device: Torch device (CPU/GPU)
        """
        self.device = device
        
        # Load model
        self.model = MultiTaskPINN(
            input_size=WINDOW_SIZE, 
            hidden_size=HIDDEN_SIZE, 
            num_classes=NUM_CLASSES
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        if 'val_velocity_mae' in checkpoint:
            print(f"   Velocity MAE: {checkpoint['val_velocity_mae']:.4f}")
        
        # Load scalers
        self._load_scalers(scalers_path)
        
        # Load video library
        self.video_library = self._load_video_library(video_library_csv)
        self.video_embeddings = None
    
    def _load_scalers(self, scalers_path):
        """Load fitted scalers from training"""
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(
                f"Scalers file not found: {scalers_path}\n"
                f"Please run training first to generate scalers!"
            )
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        self.scaler_features = scalers['features']
        self.scaler_pressure = scalers['pressure']
        self.scaler_vsg = scalers['vsg']
        self.scaler_vsl = scalers['vsl']
        
        print("\n✅ Loaded scalers from training:")
        print(f"   - Features scaler: {type(self.scaler_features).__name__}")
        print(f"   - Pressure scaler: {type(self.scaler_pressure).__name__}")
        print(f"   - Vsg scaler: {type(self.scaler_vsg).__name__}")
        print(f"   - Vsl scaler: {type(self.scaler_vsl).__name__}")
    
    def _load_video_library(self, csv_path):
        """Load video library from CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Video library CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"\n✅ Loaded video library with {len(df)} videos")
        print(f"   Flow regimes: {df['flow_regime'].value_counts().to_dict()}")
        return df
    
    def preprocess_test_data(self, file_path):
        """
        Load and preprocess test data
        
        Args:
            file_path: Path to test Excel file
            
        Returns:
            Tuple of (pressure_windows, vsg_true, vsl_true, filename)
        """
        df = pd.read_excel(file_path)
        
        # Find pressure column
        pressure_col = None
        for col in ['Pressure/bar', 'Pressure (barA)', 'Pressure']:
            if col in df.columns:
                pressure_col = col
                break
        
        if pressure_col is None:
            raise ValueError(
                f"No pressure column found in {file_path}. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        pressure = df[pressure_col].values
        filename = os.path.basename(file_path)
        vsg_true, vsl_true = extract_velocities_from_filename(filename)
        
        # Create sliding windows
        windows = []
        for i in range(0, len(pressure) - WINDOW_SIZE + 1, STRIDE):
            window = pressure[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)
        
        if len(windows) == 0:
            raise ValueError(
                f"No valid windows created from {file_path}. "
                f"Pressure length: {len(pressure)}, Window size: {WINDOW_SIZE}"
            )
        
        return np.array(windows), vsg_true, vsl_true, filename
    
    def predict_flow_regime(self, pressure_windows, vsg_input, vsl_input):
        """
        Predict flow regime and extract embeddings
        
        Args:
            pressure_windows: Array of pressure windows
            vsg_input: Superficial gas velocity
            vsl_input: Superficial liquid velocity
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features_list = [extract_features(window) for window in pressure_windows]
        features = np.array(features_list)
        
        # Scale inputs
        pressure_scaled = self.scaler_pressure.transform(pressure_windows)
        features_scaled = self.scaler_features.transform(features)
        
        # Scale velocities
        vsg_scaled = self.scaler_vsg.transform([[vsg_input]])[0]
        vsl_scaled = self.scaler_vsl.transform([[vsl_input]])[0]
        velocities_scaled = np.hstack([vsg_scaled, vsl_scaled])
        velocities_input = np.tile(velocities_scaled, (len(pressure_windows), 1))
        
        # Convert to tensors
        pressure_tensor = torch.FloatTensor(pressure_scaled).to(self.device)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        velocities_tensor = torch.FloatTensor(velocities_input).to(self.device)
        
        # Inference
        with torch.no_grad():
            class_output, velocity_output, physics_output = self.model(
                pressure_tensor, features_tensor, velocities_tensor
            )
            
            # Get embeddings from shared layer
            x = pressure_tensor.unsqueeze(1)
            x = torch.relu(self.model.bn1(self.model.conv1(x)))
            x = self.model.pool(x)
            x = torch.relu(self.model.bn2(self.model.conv2(x)))
            x = self.model.dropout(x)
            x = x.view(x.size(0), -1)
            
            feat = self.model.fc_features(features_tensor)
            vel = self.model.fc_velocities(velocities_tensor)
            combined = torch.cat([x, feat, vel], dim=1)
            embeddings = self.model.shared_layer(combined)
            
            # Get predictions
            class_probs = torch.softmax(class_output, dim=1)
            avg_class_probs = class_probs.mean(dim=0).cpu().numpy()
            predicted_class = torch.argmax(torch.tensor(avg_class_probs)).item()
            
            # Average velocity predictions
            avg_velocity = velocity_output.mean(dim=0).cpu().numpy()
            vsg_pred = self.scaler_vsg.inverse_transform(avg_velocity[0].reshape(-1, 1))[0, 0]
            vsl_pred = self.scaler_vsl.inverse_transform(avg_velocity[1].reshape(-1, 1))[0, 0]
            
            # Average embedding
            avg_embedding = embeddings.mean(dim=0).cpu().numpy()
        
        results = {
            'predicted_class': predicted_class,
            'predicted_class_name': CLASS_NAMES[predicted_class],
            'class_probabilities': avg_class_probs,
            'predicted_vsg': vsg_pred,
            'predicted_vsl': vsl_pred,
            'embedding': avg_embedding
        }
        
        return results
    
    def compute_video_library_embeddings(self, use_actual_pressure=False):
        """
        Precompute embeddings for all videos in library
        
        Args:
            use_actual_pressure: Whether to use actual pressure data if available
        """
        print("\n" + "="*70)
        print("Computing video library embeddings...")
        print("="*70)
        
        embeddings = []
        
        for idx, row in self.video_library.iterrows():
            vsg = row['vsg']
            vsl = row['vsl']
            
            # Generate synthetic pressure (or load actual if available)
            if use_actual_pressure and 'pressure_file' in row:
                try:
                    pressure = self._load_pressure_from_file(row['pressure_file'])
                except:
                    pressure = self._generate_synthetic_pressure(vsg, vsl)
            else:
                pressure = self._generate_synthetic_pressure(vsg, vsl)
            
            pressure = pressure.reshape(1, -1)
            result = self.predict_flow_regime(pressure, vsg, vsl)
            embeddings.append(result['embedding'])
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(self.video_library)} videos")
        
        self.video_embeddings = np.array(embeddings)
        print(f"✅ Computed {len(embeddings)} video embeddings")
        print("="*70)
    
    def _generate_synthetic_pressure(self, vsg, vsl):
        """Generate synthetic pressure signal based on velocities"""
        t = np.linspace(0, 4*np.pi, WINDOW_SIZE)
        vm = vsg + vsl
        amplitude = 0.5 + 0.3 * vm
        frequency = 1.0 + 0.5 * vsg
        
        pressure = (amplitude * np.sin(frequency * t) + 
                   0.3 * amplitude * np.sin(2 * frequency * t) +
                   0.1 * np.random.normal(0, 1, WINDOW_SIZE))
        
        pressure += 1.0 + 0.1 * vm
        return pressure
    
    def _load_pressure_from_file(self, pressure_file):
        """Load pressure data from file"""
        df = pd.read_excel(pressure_file)
        for col in ['Pressure/bar', 'Pressure (barA)', 'Pressure']:
            if col in df.columns:
                pressure = df[col].values[:WINDOW_SIZE]
                if len(pressure) < WINDOW_SIZE:
                    pressure = np.pad(pressure, (0, WINDOW_SIZE - len(pressure)), mode='edge')
                return pressure
        raise ValueError(f"No pressure column found in {pressure_file}")
    
    def retrieve_similar_videos(self, query_embedding, query_class, top_k=TOP_K, 
                                similarity_metric='cosine'):
        """
        Retrieve top-k similar videos from the same flow regime
        
        Args:
            query_embedding: Query embedding vector
            query_class: Predicted flow regime class
            top_k: Number of videos to retrieve
            similarity_metric: 'cosine' or 'euclidean'
            
        Returns:
            DataFrame of top-k similar videos
        """
        # Filter by flow regime
        same_regime_videos = self.video_library[
            self.video_library['flow_regime_idx'] == query_class
        ].copy()
        
        if len(same_regime_videos) == 0:
            print(f"⚠️  No videos found for flow regime: {CLASS_NAMES[query_class]}")
            return pd.DataFrame()
        
        # Get embeddings for same regime
        same_regime_indices = same_regime_videos.index.tolist()
        same_regime_embeddings = self.video_embeddings[same_regime_indices]
        
        # Calculate similarities
        if similarity_metric == 'cosine':
            similarities = [
                cosine_similarity(query_embedding, video_emb) 
                for video_emb in same_regime_embeddings
            ]
        else:  # euclidean
            similarities = [
                -euclidean_distance(query_embedding, video_emb)
                for video_emb in same_regime_embeddings
            ]
        
        same_regime_videos['similarity_score'] = similarities
        
        # Get top-k
        top_videos = same_regime_videos.nlargest(
            min(top_k, len(same_regime_videos)), 
            'similarity_score'
        )
        
        return top_videos
    
    def visualize_results(self, test_file, results, retrieved_videos):
        """Visualize retrieval results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Classification probabilities
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if i == results['predicted_class'] else '#3498db' 
                  for i in range(NUM_CLASSES)]
        bars = ax1.bar(CLASS_NAMES, results['class_probabilities'], color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_title('Flow Regime Classification Probabilities', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        
        for bar, prob in zip(bars, results['class_probabilities']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Velocity predictions
        ax2 = axes[0, 1]
        velocities = ['Vsg (m/s)', 'Vsl (m/s)']
        predicted_vals = [results['predicted_vsg'], results['predicted_vsl']]
        bars = ax2.bar(velocities, predicted_vals, color=['#e74c3c', '#9b59b6'], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
        ax2.set_title('Predicted Superficial Velocities', 
                     fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, predicted_vals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. Retrieved videos similarity scores
        ax3 = axes[1, 0]
        if len(retrieved_videos) > 0:
            video_labels = [f"Video {i+1}" for i in range(len(retrieved_videos))]
            similarity_scores = retrieved_videos['similarity_score'].values
            bars = ax3.barh(video_labels, similarity_scores, color='#1abc9c', 
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
            ax3.set_title(f'Top-{len(retrieved_videos)} Retrieved Videos', 
                         fontsize=14, fontweight='bold')
            
            for bar, score in zip(bars, similarity_scores):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2.,
                        f' {score:.4f}', ha='left', va='center', 
                        fontweight='bold', fontsize=10)
            ax3.grid(axis='x', alpha=0.3, linestyle='--')
        else:
            ax3.text(0.5, 0.5, 'No videos retrieved', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='red')
            ax3.set_xlim([0, 1])
            ax3.axis('off')
        
        # 4. Retrieved videos info
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if len(retrieved_videos) > 0:
            info_text = f"Test File:\n{os.path.basename(test_file)}\n\n"
            info_text += f"Predicted: {results['predicted_class_name']}\n"
            info_text += f"Confidence: {results['class_probabilities'][results['predicted_class']]:.3f}\n\n"
            info_text += f"Predicted Vsg: {results['predicted_vsg']:.3f} m/s\n"
            info_text += f"Predicted Vsl: {results['predicted_vsl']:.3f} m/s\n\n"
            info_text += "="*45 + "\n"
            info_text += "Retrieved Videos:\n"
            info_text += "="*45 + "\n"
            
            for idx, (_, row) in enumerate(retrieved_videos.iterrows(), 1):
                info_text += f"\n{idx}. {row['filename']}\n"
                info_text += f"   Vsg={row['vsg']:.3f}, Vsl={row['vsl']:.3f}\n"
                info_text += f"   Similarity: {row['similarity_score']:.4f}\n"
        else:
            info_text = "No videos retrieved"
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                         edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        output_file = f"retrieval_results_{os.path.basename(test_file).replace('.xlsx', '.png')}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {output_file}")
        plt.show()
    
    def extract_video_frames(self, video_path, output_dir=None, num_frames=6):
        """Extract representative frames from video"""
        if output_dir is None:
            output_dir = "video_frames"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📸 Extracting {num_frames} frames from video...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        saved_frames = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                output_path = os.path.join(output_dir, f"{video_name}_frame_{idx+1}.png")
                cv2.imwrite(output_path, frame)
                saved_frames.append(output_path)
                print(f"   Saved: {output_path}")
        
        cap.release()
        print(f"✅ Extracted {len(saved_frames)} frames")
        
        if len(saved_frames) > 0:
            self._display_frame_grid(saved_frames)
        
        return saved_frames
    
    def _display_frame_grid(self, frame_paths):
        """Display extracted frames in a grid"""
        n_frames = len(frame_paths)
        if n_frames == 0:
            return
        
        cols = min(3, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for idx, frame_path in enumerate(frame_paths):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ax.imshow(frame_rgb)
            ax.set_title(f"Frame {idx+1}", fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_frames, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('video_frame_grid.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Frame grid saved as 'video_frame_grid.png'")
        plt.show()
    
    def run_retrieval(self, test_file_path, top_k=TOP_K, show_frames=True):
        """
        Main retrieval pipeline
        
        Args:
            test_file_path: Path to test file
            top_k: Number of videos to retrieve
            show_frames: Whether to extract and show video frames
            
        Returns:
            Tuple of (results, retrieved_videos)
        """
        print("\n" + "="*70)
        print(f"PROCESSING TEST FILE")
        print("="*70)
        print(f"File: {os.path.basename(test_file_path)}")
        
        try:
            # Preprocess test data
            pressure_windows, vsg_true, vsl_true, filename = \
                self.preprocess_test_data(test_file_path)
            
            print(f"\n📊 Test data loaded:")
            print(f"   Number of windows: {len(pressure_windows)}")
            print(f"   True Vsg: {vsg_true:.3f} m/s")
            print(f"   True Vsl: {vsl_true:.3f} m/s")
            
            # Run inference
            print("\n🔮 Running inference...")
            results = self.predict_flow_regime(pressure_windows, vsg_true, vsl_true)
            
            print(f"\n📈 Prediction Results:")
            print(f"   Predicted Flow Regime: {results['predicted_class_name']}")
            print(f"   Confidence: {results['class_probabilities'][results['predicted_class']]:.3f}")
            print(f"   Predicted Vsg: {results['predicted_vsg']:.3f} m/s (True: {vsg_true:.3f})")
            print(f"   Predicted Vsl: {results['predicted_vsl']:.3f} m/s (True: {vsl_true:.3f})")
            
            # Calculate errors
            vsg_error = abs(results['predicted_vsg'] - vsg_true)
            vsl_error = abs(results['predicted_vsl'] - vsl_true)
            print(f"\n📉 Prediction Errors:")
            print(f"   Vsg Error: {vsg_error:.4f} m/s ({100*vsg_error/max(vsg_true, 0.01):.2f}%)")
            print(f"   Vsl Error: {vsl_error:.4f} m/s ({100*vsl_error/max(vsl_true, 0.01):.2f}%)")
            
            # Retrieve similar videos
            print(f"\n🔍 Retrieving top-{top_k} similar videos...")
            retrieved_videos = self.retrieve_similar_videos(
                results['embedding'], 
                results['predicted_class'], 
                top_k=top_k
            )
            
            if len(retrieved_videos) > 0:
                print(f"\n✅ Top-{len(retrieved_videos)} Retrieved Videos:")
                print("-" * 70)
                for idx, (_, row) in enumerate(retrieved_videos.iterrows(), 1):
                    print(f"{idx}. {row['filename']}")
                    print(f"   Vsg={row['vsg']:.3f} m/s, Vsl={row['vsl']:.3f} m/s")
                    print(f"   Similarity Score: {row['similarity_score']:.4f}")
                    if 'video_path' in row:
                        print(f"   Path: {row['video_path']}")
                    print()
                
                # Extract frames from top video if requested
                if show_frames and 'video_path' in retrieved_videos.columns:
                    top_video_path = retrieved_videos.iloc[0]['video_path']
                    if os.path.exists(top_video_path):
                        print(f"\n{'='*70}")
                        print("🎬 Top Retrieved Video")
                        print(f"{'='*70}")
                        self.extract_video_frames(top_video_path, num_frames=6)
                    else:
                        print(f"\n⚠️  Top video file not found: {top_video_path}")
            else:
                print("\n⚠️  No similar videos found!")
            
            # Visualize results
            print("\n📊 Generating visualization...")
            self.visualize_results(test_file_path, results, retrieved_videos)
            
            print("="*70)
            return results, retrieved_videos
            
        except Exception as e:
            print(f"\n❌ Error processing test file: {e}")
            import traceback
            traceback.print_exc()
            return None, None