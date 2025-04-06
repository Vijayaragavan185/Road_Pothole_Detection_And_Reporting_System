from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import numpy as np
import openvino as ov
from datetime import datetime
import os

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

# Add some test data
def add_test_data():
    conn = sqlite3.connect('potholes.db')
    cursor = conn.cursor()
    # Check if we already have data
    cursor.execute("SELECT COUNT(*) FROM potholes")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Add some sample data
        test_data = [
            (20.5937, 78.9629, 0.85, datetime.now().isoformat()),  # Center of India
            (20.6037, 78.9729, 0.65, datetime.now().isoformat()),  # Nearby
            (20.5837, 78.9529, 0.95, datetime.now().isoformat()),  # Another nearby
        ]
        cursor.executemany(
            "INSERT INTO potholes (latitude, longitude, severity, timestamp) VALUES (?, ?, ?, ?)",
            test_data
        )
        conn.commit()
        print("Added test pothole data")
    
    conn.close()

# Root route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

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
    # This is a placeholder - implement your actual feature extraction
    # It should match the feature extraction in your ESP32 code (extract_features function)
    features = np.zeros(34, dtype=np.float32)  # Assuming 34 features based on model
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
    add_test_data()  # Add test data for demonstration
    app.run(debug=True, host='0.0.0.0', port=5000)
