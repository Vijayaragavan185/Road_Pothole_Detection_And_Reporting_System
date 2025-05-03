import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

# Load the SVM model specifically
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Determine the number of features the model expects
n_features = svm_model.n_features_in_

# Generate synthetic data with the correct number of features
n_samples = 200
np.random.seed(42)
X_synthetic = np.random.randn(n_samples, n_features)

# Create labels based on the model's decision function
y_synthetic = svm_model.predict(X_synthetic)

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_synthetic)

# Get support vectors and transform them with PCA
support_vectors = svm_model.support_vectors_
support_vectors_pca = pca.transform(support_vectors)

# Create a mesh grid for decision boundary visualization
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Map the mesh grid back to the original feature space
Z_pca = np.c_[xx.ravel(), yy.ravel()]
Z_orig = pca.inverse_transform(Z_pca)

# Predict on the mesh grid
Z = svm_model.predict(Z_orig)
Z = Z.reshape(xx.shape)

# Calculate decision function for margin visualization
decision_function = svm_model.decision_function(Z_orig).reshape(xx.shape)

# Plot the visualization
plt.figure(figsize=(12, 9))

# Plot the decision boundary
contourf = plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot data points
pothole_indices = y_synthetic == 1
non_pothole_indices = y_synthetic == 0
plt.scatter(X_pca[non_pothole_indices, 0], X_pca[non_pothole_indices, 1], 
            c='blue', s=60, edgecolor='k', label='Non-pothole')
plt.scatter(X_pca[pothole_indices, 0], X_pca[pothole_indices, 1], 
            c='red', s=60, edgecolor='k', label='Pothole')

# Highlight support vectors
plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1], 
            s=150, facecolors='none', edgecolors='green', linewidth=2,
            label='Support Vectors')

# Add decision boundary contour
plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0.5])

# Add margins
plt.contour(xx, yy, decision_function, colors=['k'], linestyles=['--'], levels=[-1, 1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Hyperplane and Support Vectors for Pothole Detection')
plt.legend()
plt.colorbar(contourf, label='Decision Function')
plt.tight_layout()
plt.savefig('svm_visualization.png', dpi=300)
plt.show()
