import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class RoadDataAnalyzer:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf')
        
    def load_data(self, file_path):
        """Load the CSV data file"""
        self.data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.data)} samples")
        return self.data
    
    def plot_segment(self, start_idx=0, duration=5):
        """Plot a segment of the data with pothole labels"""
        samples = int(duration * 100)  # Assuming 100Hz sampling rate
        end_idx = start_idx + samples
        
        plt.figure(figsize=(15, 10))
        
        # Plot accelerometer data
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            plt.subplot(4, 1, i+1)
            plt.plot(self.data['timestamp'][start_idx:end_idx], 
                    self.data[axis][start_idx:end_idx])
            plt.title(f'{axes[i]}-axis Acceleration')
            plt.ylabel('g')
            
            # Highlight pothole regions
            self._highlight_potholes(start_idx, end_idx)
        
        # Plot labels
        plt.subplot(4, 1, 4)
        plt.plot(self.data['timestamp'][start_idx:end_idx], 
                self.data['label'][start_idx:end_idx])
        plt.title('Pothole Labels')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()
    
    def _highlight_potholes(self, start_idx, end_idx):
        """Highlight regions where potholes occur"""
        pothole_regions = self.data['label'][start_idx:end_idx] == 1
        if pothole_regions.any():
            plt.fill_between(self.data['timestamp'][start_idx:end_idx], 
                           plt.ylim()[0], plt.ylim()[1],
                           where=pothole_regions,
                           color='red', alpha=0.3)
    
    def extract_features(self, window):
        """Extract features from a window of sensor data"""
        features = {}
        
        # Time domain features
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            axis_data = window[axis]
            features.update({
                f'{axis}_mean': np.mean(axis_data),
                f'{axis}_std': np.std(axis_data),
                f'{axis}_max': np.max(axis_data),
                f'{axis}_min': np.min(axis_data),
                f'{axis}_peak2peak': np.max(axis_data) - np.min(axis_data),
                f'{axis}_rms': np.sqrt(np.mean(np.square(axis_data)))
            })
            
            # Frequency domain features
            fft = np.fft.fft(axis_data)
            fft_magnitude = np.abs(fft)
            features.update({
                f'{axis}_fft_mean': np.mean(fft_magnitude),
                f'{axis}_fft_max': np.max(fft_magnitude)
            })
        
        return features
    
    def prepare_training_data(self):
        """Prepare windowed data for training"""
        windows = []
        labels = []
        
        for i in range(0, len(self.data) - self.window_size, self.window_size // 2):
            window = self.data.iloc[i:i + self.window_size]
            windows.append(self.extract_features(window))
            # Use majority vote for window label
            labels.append(int(window['label'].mean() > 0.5))
        
        # Convert to DataFrame
        X = pd.DataFrame(windows)
        y = np.array(labels)
        
        return X, y
    
    def train_model(self):
        """Train the pothole detection model"""
        print("Preparing training data...")
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_test, y_pred

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = RoadDataAnalyzer()
    
    # Load the synthetic data
    data = analyzer.load_data('synthetic_road_data.csv')
    
    # Plot first 5 seconds of data
    analyzer.plot_segment(duration=5)
    
    # Train and evaluate model
    y_test, y_pred = analyzer.train_model()