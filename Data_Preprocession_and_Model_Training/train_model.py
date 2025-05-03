import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
            features[f'{axis}_range'] = window[axis].max() - window[axis].min()
            
            # Add jerk features if available
            if f'{axis}_jerk' in window.columns:
                features[f'{axis}_jerk_mean'] = window[f'{axis}_jerk'].mean()
                features[f'{axis}_jerk_max'] = window[f'{axis}_jerk'].max()
                features[f'{axis}_jerk_std'] = window[f'{axis}_jerk'].std()
        
        # Magnitude features
        features['acc_mag1_max'] = window['acc_mag1'].max()
        features['acc_mag2_max'] = window['acc_mag2'].max()
        features['acc_mag1_mean'] = window['acc_mag1'].mean()
        features['acc_mag2_mean'] = window['acc_mag2'].mean()
        features['acc_mag1_std'] = window['acc_mag1'].std()
        features['acc_mag2_std'] = window['acc_mag2'].std()
        
        # Z-axis to XY-plane ratio features (if available)
        if 'z_xy_ratio1' in window.columns:
            features['z_xy_ratio1_max'] = window['z_xy_ratio1'].max()
            features['z_xy_ratio1_mean'] = window['z_xy_ratio1'].mean()
            features['z_xy_ratio2_max'] = window['z_xy_ratio2'].max()
            features['z_xy_ratio2_mean'] = window['z_xy_ratio2'].mean()
        else:
            # Calculate ratios on the fly if not already in data
            z1 = np.abs(window['acc_z1'])
            xy1 = np.abs(window['acc_x1']) + np.abs(window['acc_y1']) + 0.1
            z2 = np.abs(window['acc_z2'])
            xy2 = np.abs(window['acc_x2']) + np.abs(window['acc_y2']) + 0.1
            
            ratio1 = z1 / xy1
            ratio2 = z2 / xy2
            
            features['z_xy_ratio1_max'] = ratio1.max()
            features['z_xy_ratio1_mean'] = ratio1.mean()
            features['z_xy_ratio2_max'] = ratio2.max()
            features['z_xy_ratio2_mean'] = ratio2.mean()
        
        # Differential features if available
        for axis in ['acc_x_diff', 'acc_y_diff', 'acc_z_diff']:
            if axis in window.columns:
                features[f'{axis}_mean'] = window[axis].mean()
                features[f'{axis}_max'] = window[axis].max()
                features[f'{axis}_std'] = window[axis].std()
        
        # Gyroscope features
        for axis in ['gyr_x1', 'gyr_y1', 'gyr_z1', 'gyr_x2', 'gyr_y2', 'gyr_z2']:
            features[f'{axis}_std'] = window[axis].std()
            features[f'{axis}_mean'] = window[axis].mean()
            features[f'{axis}_max'] = np.abs(window[axis]).max()
            
        # Add gyro magnitude if available
        if 'gyr_mag1' in window.columns:
            features['gyr_mag1_mean'] = window['gyr_mag1'].mean()
            features['gyr_mag2_mean'] = window['gyr_mag2'].mean()
            features['gyr_mag1_max'] = window['gyr_mag1'].max()
            features['gyr_mag2_max'] = window['gyr_mag2'].max()
        else:
            # Calculate gyro magnitudes on the fly
            gyr_mag1 = np.sqrt(window['gyr_x1']**2 + window['gyr_y1']**2 + window['gyr_z1']**2)
            gyr_mag2 = np.sqrt(window['gyr_x2']**2 + window['gyr_y2']**2 + window['gyr_z2']**2)
            features['gyr_mag1_mean'] = gyr_mag1.mean()
            features['gyr_mag2_mean'] = gyr_mag2.mean()
            features['gyr_mag1_max'] = gyr_mag1.max()
            features['gyr_mag2_max'] = gyr_mag2.max()
        
        # Z-score features if available
        for col in ['acc_z1_zscore', 'acc_z2_zscore', 'acc_mag1_zscore', 'acc_mag2_zscore']:
            if col in window.columns:
                features[f'{col}_max'] = window[col].max()
                features[f'{col}_min'] = window[col].min()
                features[f'{col}_std'] = window[col].std()
        
        windows.append(features)
        
        # Label: if any point in window is pothole, label as pothole
        has_pothole = False
        if 'auto_pothole' in window.columns:
            has_pothole = window['auto_pothole'].sum() > 0
        elif 'is_pothole' in window.columns:
            has_pothole = window['is_pothole'].sum() > 0
            
        labels.append(1 if has_pothole else 0)
    
    return pd.DataFrame(windows), np.array(labels)

def visualize_model(X_train_scaled, y_train, model, output_file='svm_visualization.png'):
    """Visualize the SVM model using PCA for dimensionality reduction"""
    # Use PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Create a mesh to plot the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Train SVM on PCA data for visualization
    svm_pca = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma if hasattr(model, 'gamma') else 'auto')
    svm_pca.fit(X_train_pca, y_train)
    
    # Plot the decision boundary
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    # Plot the training points
    plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], 
                c='blue', marker='o', edgecolors='k', label='Non-pothole')
    plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], 
                c='red', marker='o', edgecolors='k', label='Pothole')
    
    # Plot support vectors
    plt.scatter(svm_pca.support_vectors_[:, 0], svm_pca.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='green', label='Support Vectors')
    
    plt.title(f'{model.kernel.capitalize()} Kernel SVM Decision Boundary')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.colorbar(label='Decision Function')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

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
            pothole_rows = data[data['auto_pothole'] == 1] if 'auto_pothole' in data.columns else data[data['is_pothole'] == 1]
            non_pothole_column = 'auto_pothole' if 'auto_pothole' in data.columns else 'is_pothole'
            non_pothole_rows = data[data[non_pothole_column] == 0].sample(
                min(len(pothole_rows)*3, len(data[data[non_pothole_column] == 0])), 
                random_state=42)
            phase_data = pd.concat([pothole_rows, non_pothole_rows])
        else:
            # Sample to keep dataset sizes reasonable
            if len(data) > 10000:
                phase_data = data.sample(10000, random_state=42)
            else:
                phase_data = data
        
        all_data.append(phase_data)
    
    # Combine all data
    combined_data = pd.concat(all_data)
    print(f"Combined dataset has {len(combined_data)} samples")
    
    # Create windows and extract features
    X, y = create_windows(combined_data)
    print(f"Created {len(X)} feature windows with {sum(y)} pothole windows")
    
    # Check for NaN values
    print("Checking for NaN values...")
    nan_count_before = X.isna().sum().sum()
    if nan_count_before > 0:
        print(f"Found {nan_count_before} NaN values in features. Filling with zeros...")
        X = X.fillna(0)  # Replace NaN with zeros
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    try:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), 
                                  param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    except Exception as e:
        print(f"Grid search failed: {e}")
        print("Falling back to default SVM model...")
        model = SVC(kernel='linear', probability=True, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize model
    print("Creating model visualization...")
    visualize_model(X_train_scaled, y_train, model)
    
    # Extract model parameters for ESP32
    if model.kernel == 'linear':
        weights = model.coef_[0]
        bias = model.intercept_[0]
        
        print("\nModel Parameters for ESP32:")
        print(f"const int NUM_FEATURES = {len(weights)};")
        print("const float SVM_WEIGHTS[] = {")
        for i, w in enumerate(weights):
            print(f" {w:.6f}", end=",\n" if i < len(weights)-1 else "\n")
        print("};")
        print(f"const float SVM_BIAS = {bias:.6f};")
        
        # Generate C++ header file
        with open('model_params.h', 'w') as f:
            f.write("// Auto-generated model parameters\n\n")
            f.write(f"const int NUM_FEATURES = {len(weights)};\n\n")
            f.write("const float SVM_WEIGHTS[] = {\n")
            for i, w in enumerate(weights):
                f.write(f" {w:.6f}" + ("," if i < len(weights)-1 else "") + "\n")
            f.write("};\n\n")
            f.write(f"const float SVM_BIAS = {bias:.6f};\n")
            f.write("const float SVM_THRESHOLD = 0.0;\n")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    with open('pothole_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names for reference
    with open('feature_names.txt', 'w') as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()
