import pickle
import numpy as np
import openvino as ov

def convert_svm_to_ov_directly():
    # Load trained SVM model
    with open('pothole_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    
    # Get model parameters
    weights = svm.coef_[0]
    bias = svm.intercept_[0]
    num_features = len(weights)
    
    # Create a simple XML representation of the model
    xml_content = f"""<?xml version="1.0" ?>
<net name="SVM" version="10">
    <layers>
        <layer id="0" name="input" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,{num_features}"/>
            <output>
                <port id="0" precision="FP32"/>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="{num_features},1" size="{4*num_features}"/>
            <output>
                <port id="1" precision="FP32"/>
            </output>
        </layer>
        <layer id="2" name="MatMul" type="MatMul" version="opset1">
            <input>
                <port id="0"/>
                <port id="1"/>
            </input>
            <output>
                <port id="2" precision="FP32"/>
            </output>
        </layer>
        <layer id="3" name="bias" type="Const" version="opset1">
            <data element_type="f32" offset="{4*num_features}" shape="1,1" size="4"/>
            <output>
                <port id="3" precision="FP32"/>
            </output>
        </layer>
        <layer id="4" name="Add" type="Add" version="opset1">
            <input>
                <port id="0"/>
                <port id="1"/>
            </input>
            <output>
                <port id="2" precision="FP32"/>
            </output>
        </layer>
        <layer id="5" name="Sigmoid" type="Sigmoid" version="opset1">
            <input>
                <port id="0"/>
            </input>
            <output>
                <port id="1" precision="FP32"/>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
"""
    
    # Write XML to file
    with open("pothole_ov_model.xml", "w") as f:
        f.write(xml_content)
    
    # Create binary file with weights and bias
    with open("pothole_ov_model.bin", "wb") as f:
        f.write(weights.astype(np.float32).tobytes())
        f.write(np.array([bias], dtype=np.float32).tobytes())
    
    print("Model conversion completed via direct XML writing")

if __name__ == "__main__":
    convert_svm_to_ov_directly()
