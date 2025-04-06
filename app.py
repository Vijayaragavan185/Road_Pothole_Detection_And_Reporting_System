from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import numpy as np
import openvino as ov
from datetime import datetime
import math

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
        # Add some sample data for different regions in India
        test_data = [
            # Chennai area
            (13.0827, 80.2707, 0.85, datetime.now().isoformat()),  # Chennai
            (13.1067, 80.2847, 0.65, datetime.now().isoformat()),  # Nearby
            (13.0647, 80.2567, 0.95, datetime.now().isoformat()),  # Another nearby
            
            # Chengalpattu area
            (12.6819, 79.9888, 0.90, datetime.now().isoformat()),  # Chengalpattu
            (12.6919, 79.9988, 0.75, datetime.now().isoformat()),  # Nearby
            (12.6719, 79.9788, 0.85, datetime.now().isoformat()),  # Another nearby
            
            # Route between Chennai and Chengalpattu
            (12.8823, 80.1353, 0.78, datetime.now().isoformat()),  # Midway point
            (12.9412, 80.1830, 0.88, datetime.now().isoformat()),  # Another on route
            (12.7631, 80.0576, 0.92, datetime.now().isoformat()),  # Another on route
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
@app.route('/api/route', methods=['GET'])
def get_route_info():
    start_lat = float(request.args.get('start_lat'))
    start_lng = float(request.args.get('start_lng'))
    end_lat = float(request.args.get('end_lat'))
    end_lng = float(request.args.get('end_lng'))
    
    # Query potholes within the route bounds
    conn = sqlite3.connect('potholes.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM potholes 
        WHERE latitude BETWEEN ? AND ? 
        AND longitude BETWEEN ? AND ?
    """, (start_lat, end_lat, start_lng, end_lng))
    pothole_count = cursor.fetchone()[0]
    
    # Get the potholes along the route for highlighting
    cursor.execute("""
        SELECT latitude, longitude, severity, timestamp FROM potholes 
        WHERE latitude BETWEEN ? AND ? 
        AND longitude BETWEEN ? AND ?
    """, (start_lat, end_lat, start_lng, end_lng))
    
    potholes = [
        {"lat": row[0], "lng": row[1], "severity": row[2], "timestamp": row[3]} 
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    return jsonify({
        "pothole_count": pothole_count,
        "potholes": potholes
    })


def calculate_distance(lat1, lng1, lat2, lng2):
    # Haversine formula to calculate distance between coordinates
    R = 6371  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat/2) * math.sin(d_lat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(d_lng/2) * math.sin(d_lng/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def extract_features(data):
    # Implement feature extraction similar to your create_windows() function
    # This should match the 34 features from your SVM model
    features = np.zeros(34, dtype=np.float32)
    
    # Process accelerometer data
    if 'accelerometer_data' in data:
        acc_data = data['accelerometer_data']
        
        # Extract features based on your create_windows() function
        # Time domain features for each axis
        axes = ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']
        feat_idx = 0
        
        for axis in axes:
            values = [sample[axis] for sample in acc_data]
            features[feat_idx] = min(values)  # min
            features[feat_idx+1] = max(values)  # max
            features[feat_idx+2] = sum(values) / len(values)  # mean
            features[feat_idx+3] = np.std(values)  # std
            feat_idx += 4
        
        # Magnitude features
        mag1_values = [np.sqrt(sample['acc_x1']**2 + sample['acc_y1']**2 + sample['acc_z1']**2) 
                     for sample in acc_data]
        mag2_values = [np.sqrt(sample['acc_x2']**2 + sample['acc_y2']**2 + sample['acc_z2']**2) 
                     for sample in acc_data]
        
        features[feat_idx] = max(mag1_values)  # mag1_max
        features[feat_idx+1] = max(mag2_values)  # mag2_max
        features[feat_idx+2] = np.std(mag1_values)  # mag1_std
        features[feat_idx+3] = np.std(mag2_values)  # mag2_std
        feat_idx += 4
        
        # Gyroscope features
        gyro_axes = ['gyr_x1', 'gyr_y1', 'gyr_z1', 'gyr_x2', 'gyr_y2', 'gyr_z2']
        for axis in gyro_axes:
            values = [sample[axis] for sample in acc_data]
            features[feat_idx] = np.std(values)  # std
            feat_idx += 1
    
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
