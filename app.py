from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import numpy as np
import openvino as ov
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize OpenVINO
core = ov.Core()
model = core.read_model("pothole_ov_model.xml")
compiled_model = core.compile_model(model, "CPU")

# Create database
def init_db():
    conn = sqlite3.connect('potholes.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS potholes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        severity REAL NOT NULL,
        timestamp TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# API endpoint to receive pothole data from ESP32
@app.route('/api/detect', methods=['POST'])
def detect_pothole():
    data = request.json
    
    # Extract features from accelerometer data
    features = extract_features(data)
    
    # Run inference with OpenVINO
    input_tensor = np.array([features], dtype=np.float32)
    result = compiled_model(input_tensor)[0][0][0]
    
    is_pothole = bool(result > 0.5)
    if is_pothole:
        # Store pothole in database
        store_pothole(data['latitude'], data['longitude'], float(result))
        
    return jsonify({
        "is_pothole": is_pothole,
        "confidence": float(result),
        "timestamp": datetime.now().isoformat()
    })

# API endpoint to get all potholes
@app.route('/api/potholes', methods=['GET'])
def get_potholes():
    conn = sqlite3.connect('potholes.db')
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude, severity, timestamp FROM potholes")
    potholes = [
        {"lat": row[0], "lng": row[1], "severity": row[2], "timestamp": row[3]} 
        for row in cursor.fetchall()
    ]
    conn.close()
    return jsonify(potholes)

def extract_features(data):
    # Implement feature extraction based on your create_windows() function
    features = []
    
    # Add your feature extraction code here similar to final.c
    # This should match the 34 features from your SVM model
    
    return features

def store_pothole(lat, lng, severity):
    conn = sqlite3.connect('potholes.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO potholes (latitude, longitude, severity, timestamp) VALUES (?, ?, ?, ?)",
        (lat, lng, severity, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
