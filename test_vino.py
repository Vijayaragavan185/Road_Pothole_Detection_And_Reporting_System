import numpy as np
import openvino as ov

# Initialize OpenVINO runtime
core = ov.Core()

# Load and compile the model
model = core.read_model("pothole_detection_model.xml")
compiled_model = core.compile_model(model, "CPU")

# Create a sample feature vector (with the correct number of features)
# Replace 28 with the actual number of features from your SVM model
features = np.random.rand(1, 28).astype(np.float32)

# Run inference
results = compiled_model(features)
score = results[0][0][0]

print(f"Inference result: {score}")
print(f"Prediction: {'Pothole' if score > 0.5 else 'No Pothole'}")
