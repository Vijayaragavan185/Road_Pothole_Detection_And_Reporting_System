import geopy.distance
import sqlite3
from datetime import datetime

# Step 1: Define the start and end coordinates
start_coords = (12.985330555555556, 79.96983055555556)  # Converted from 12° 59' 7.19" N, 79° 58' 11.39" E
end_coords = (12.794378262825573, 80.03835002861089)

# Step 2: Calculate 10 evenly spaced points along the route
num_points = 10
latitudes = [start_coords[0] - i * (start_coords[0] - end_coords[0]) / (num_points - 1) for i in range(num_points)]
longitudes = [start_coords[1] + i * (end_coords[1] - start_coords[1]) / (num_points - 1) for i in range(num_points)]

# Combine coordinates into pothole locations
pothole_locations = list(zip(latitudes, longitudes))

# Step 3: Create the database table if it doesn't exist and insert the data
conn = sqlite3.connect('potholes.db')
cursor = conn.cursor()

# Create the potholes table
cursor.execute('''
CREATE TABLE IF NOT EXISTS potholes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    severity REAL NOT NULL,
    timestamp TEXT NOT NULL
)
''')

# Insert the 10 pothole locations with varying severity
for i, (lat, lng) in enumerate(pothole_locations):
    # Vary the severity slightly for more realistic data
    severity = 0.7 + (i % 3) * 0.1  # Values between 0.7 and 0.9
    
    cursor.execute(
        "INSERT INTO potholes (latitude, longitude, severity, timestamp) VALUES (?, ?, ?, ?)",
        (lat, lng, severity, datetime.now().isoformat())
    )

# Commit and close connection
conn.commit()
conn.close()

print("Sample data added successfully!")
