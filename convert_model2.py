import pickle
import numpy as np
import tensorflow as tf
import openvino as ov
import tf2onnx
import onnx

def convert_svm_to_ov():
    # Load trained SVM model
    with open('pothole_model.pkl', 'rb') as f:
        svm = pickle.load(f)

    # Create equivalent neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(svm.coef_[0]),)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Transfer SVM parameters
    weights = svm.coef_[0].reshape(-1, 1)
    bias = svm.intercept_[0]
    model.layers[0].set_weights([weights, np.array([bias])])
    
    # Convert directly to ONNX format
    input_signature = [tf.TensorSpec([None, len(svm.coef_[0])], tf.float32)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    
    # Save ONNX model
    onnx.save(onnx_model, "pothole_model.onnx")
    
    # Convert ONNX to OpenVINO
    ov_model = ov.convert_model("pothole_model.onnx")
    ov.save_model(ov_model, "pothole_ov_model.xml")

if __name__ == "__main__":
    convert_svm_to_ov()
    print("Model conversion successful! Files created: pothole_ov_model.xml, pothole_ov_model.bin")
