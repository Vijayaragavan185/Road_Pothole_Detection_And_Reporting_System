import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
processed_pothole_data = pd.read_csv("processed_pothole_data.csv")

# Function to calculate statistical features
def extract_statistical_features(data):
    """Extract statistical features from sensor data"""
    features = {}
    
    # Basic sensor columns
    sensor_cols = ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2', 
                  'gyr_x1', 'gyr_y1', 'gyr_z1', 'gyr_x2', 'gyr_y2', 'gyr_z2']
    
    for col in sensor_cols:
        # Statistical features
        features[f'{col}_mean'] = data[col].mean()
        features[f'{col}_std'] = data[col].std()
        features[f'{col}_var'] = data[col].var()
        features[f'{col}_kurtosis'] = data[col].kurtosis()
        features[f'{col}_skew'] = data[col].skew()
        features[f'{col}_max'] = data[col].max()
        features[f'{col}_min'] = data[col].min()
        features[f'{col}_range'] = data[col].max() - data[col].min()
        features[f'{col}_median'] = data[col].median()
        
        # Frequency domain features
        fft_values = np.abs(fft(data[col].values))
        features[f'{col}_fft_mean'] = np.mean(fft_values)
        features[f'{col}_fft_std'] = np.std(fft_values)
        features[f'{col}_fft_max'] = np.max(fft_values)
        features[f'{col}_fft_energy'] = np.sum(fft_values**2) / len(fft_values)
    
    # Calculate cross-sensor features
    # Accelerometer magnitude
    data['acc1_mag'] = np.sqrt(data['acc_x1']**2 + data['acc_y1']**2 + data['acc_z1']**2)
    data['acc2_mag'] = np.sqrt(data['acc_x2']**2 + data['acc_y2']**2 + data['acc_z2']**2)
    
    # Gyroscope magnitude
    data['gyr1_mag'] = np.sqrt(data['gyr_x1']**2 + data['gyr_y1']**2 + data['gyr_z1']**2)
    data['gyr2_mag'] = np.sqrt(data['gyr_x2']**2 + data['gyr_y2']**2 + data['gyr_z2']**2)
    
    # Add magnitude features
    for col in ['acc1_mag', 'acc2_mag', 'gyr1_mag', 'gyr2_mag']:
        features[f'{col}_mean'] = data[col].mean()
        features[f'{col}_std'] = data[col].std()
        features[f'{col}_max'] = data[col].max()
        features[f'{col}_min'] = data[col].min()
    
    # Sensor differences between the two accelerometers/gyroscopes
    for axis in ['x', 'y', 'z']:
        data[f'acc_{axis}_diff'] = data[f'acc_{axis}1'] - data[f'acc_{axis}2']
        data[f'gyr_{axis}_diff'] = data[f'gyr_{axis}1'] - data[f'gyr_{axis}2']
        
        # Add difference features
        for col in [f'acc_{axis}_diff', f'gyr_{axis}_diff']:
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_min'] = data[col].min()
    
    # Calculate jerk (derivative of acceleration)
    for col in sensor_cols:
        data[f'{col}_jerk'] = data[col].diff() / data['timestamp'].diff()
        
        # Skip first row which will be NaN
        if len(data) > 1:
            features[f'{col}_jerk_mean'] = data[f'{col}_jerk'].iloc[1:].mean()
            features[f'{col}_jerk_std'] = data[f'{col}_jerk'].iloc[1:].std()
            features[f'{col}_jerk_max'] = data[f'{col}_jerk'].iloc[1:].max()
    
    return features

# Function to process data in windows
def process_data_with_windows(data, window_size=20, overlap=0.5):
    """Process data using sliding windows with overlap"""
    step = int(window_size * (1 - overlap))
    windows = []
    labels = []
    
    # Group by is_pothole to ensure we don't mix classes in a window
    for label, group in data.groupby('is_pothole'):
        # Skip if group is too small
        if len(group) < window_size:
            continue
            
        # Create windows
        for i in range(0, len(group) - window_size + 1, step):
            window = group.iloc[i:i+window_size].copy()
            features = extract_statistical_features(window)
            windows.append(features)
            labels.append(label)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(windows)
    return feature_df, np.array(labels)

# Split the data
X_features, y = process_data_with_windows(processed_pothole_data)

# Handle any NaN values
X_features = X_features.fillna(0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different kernels
kernels = ['linear', 'rbf', 'poly']
best_model = None
best_score = 0

for kernel in kernels:
    # Define parameter grid
    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif kernel == 'rbf':
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    else:  # poly
        param_grid = {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}
    
    # Create SVM model with grid search
    svm = SVC(kernel=kernel, class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_scaled, y_train)
    
    # Check if this is the best model so far
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        print(f"New best model: {kernel} with score {best_score:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the best model
y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize the results with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Create a mesh to plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train SVM on PCA data for visualization
svm_pca = SVC(kernel=best_model.kernel, C=best_model.C, gamma=best_model.gamma if hasattr(best_model, 'gamma') else 'auto')
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

plt.title(f'{best_model.kernel.capitalize()} Kernel SVM Decision Boundary')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.colorbar(label='Decision Function')
plt.tight_layout()
plt.savefig('improved_svm_visualization.png')
plt.show()
