import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import os
import matplotlib.pyplot as plt

# Function to find all processed data files
def find_processed_files():
    """Find all processed data files"""
    data_files = []
    
    for file in os.listdir('.'):
        if file.startswith('processed_') and file.endswith('_data.csv'):
            phase = file.replace('processed_', '').replace('_data.csv', '')
            data_files.append((phase, file))
    
    return data_files

# Create windows of data for feature extraction
def create_windows(data, window_size=50, overlap=25):
    """Create windows of data for feature extraction"""
    windows = []
    labels = []
    
    for i in range(0, len(data) - window_size, overlap):
        window = data.iloc[i:i+window_size]
        
        # Extract features
        features = {}
        
        # Time domain features for each axis
        for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
            features[f'{axis}_min'] = window[axis].min()
            features[f'{axis}_max'] = window[axis].max()
            features[f'{axis}_mean'] = window[axis].mean()
            features[f'{axis}_std'] = window[axis].std()
        
        # Magnitude features
        features['acc_mag1_max'] = window['acc_mag1'].max()
        features['acc_mag2_max'] = window['acc_mag2'].max()
        features['acc_mag1_std'] = window['acc_mag1'].std()
        features['acc_mag2_std'] = window['acc_mag2'].std()
        
        # Gyroscope features (if available)
        if 'gyr_x1' in window.columns:
            for axis in ['gyr_x1', 'gyr_y1', 'gyr_z1', 'gyr_x2', 'gyr_y2', 'gyr_z2']:
                features[f'{axis}_std'] = window[axis].std()
        
        windows.append(features)
        
        # Label: if any point in window is pothole, label as pothole
        has_pothole = False
        if 'auto_pothole' in window.columns:
            has_pothole = window['auto_pothole'].sum() > 0
        elif 'is_pothole' in window.columns:
            has_pothole = window['is_pothole'].sum() > 0
        
        labels.append(1 if has_pothole else 0)
    
    return pd.DataFrame(windows), np.array(labels)

def train_model():
    # Find all processed data files
    data_files = find_processed_files()
    if not data_files:
        print("No processed data files found! Please run process_data.py first.")
        return
    
    print(f"Found {len(data_files)} processed data files for training.")
    
    # Load and combine all data
    all_data = []
    for phase, file_path in data_files:
        data = pd.read_csv(file_path)
        
        # Mark simulated and real pothole data as special
        if 'simulated_pothole' in phase or 'pothole_road' in phase:
            print(f"Using {phase} data with extra weight for pothole examples")
            # Keep all rows with potholes, sample from others
            pothole_rows = data[data['auto_pothole'] == 1]
            non_pothole_rows = data[data['auto_pothole'] == 0].sample(
                min(len(pothole_rows)*3, len(data[data['auto_pothole'] == 0])))
            phase_data = pd.concat([pothole_rows, non_pothole_rows])
        else:
            # Sample to keep dataset sizes reasonable
            if len(data) > 10000:
                phase_data = data.sample(10000)
            else:
                phase_data = data
        
        all_data.append(phase_data)
    
    # Combine all data
    combined_data = pd.concat(all_data)
    print(f"Combined dataset has {len(combined_data)} samples")
    
    # Create windows and extract features
    X, y = create_windows(combined_data)
    print(f"Created {len(X)} feature windows with {sum(y)} pothole windows")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    print("Training SVM model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Extract model parameters for ESP32
    if model.kernel == 'linear':
        weights = model.coef_[0]
        bias = model.intercept_[0]
        
        print("\nModel Parameters for ESP32:")
        print(f"const int NUM_FEATURES = {len(weights)};")
        print("const float SVM_WEIGHTS[] = {")
        for i, w in enumerate(weights):
            print(f"  {w:.6f}", end=",\n" if i < len(weights)-1 else "\n")
        print("};")
        print(f"const float SVM_BIAS = {bias:.6f};")
        
        # Generate C++ header file
        with open('model_params.h', 'w') as f:
            f.write("// Auto-generated model parameters\n\n")
            f.write(f"const int NUM_FEATURES = {len(weights)};\n\n")
            f.write("const float SVM_WEIGHTS[] = {\n")
            for i, w in enumerate(weights):
                f.write(f"  {w:.6f}" + ("," if i < len(weights)-1 else "") + "\n")
            f.write("};\n\n")
            f.write(f"const float SVM_BIAS = {bias:.6f};\n")
            f.write("const float SVM_THRESHOLD = 0.0;\n")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    with open('pothole_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()