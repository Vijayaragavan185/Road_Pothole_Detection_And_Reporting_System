import serial
import time
import serial.tools.list_ports
import json
import re

# Collection settings
COLLECTION_PHASE = "smooth_road"
COLLECTION_TIME = 180  # 3 minutes

FILENAME = f"{COLLECTION_PHASE}_data.csv"
CSV_HEADER = "timestamp,acc_x1,acc_y1,acc_z1,acc_x2,acc_y2,acc_z2,gyr_x1,gyr_y1,gyr_z1,gyr_x2,gyr_y2,gyr_z2,is_pothole"

print("RIDE ON SMOOTH ROAD for 3 minutes during recording!")


# Find available ports
ports = list(serial.tools.list_ports.comports())
print("Available COM ports:")
for p in ports:
    print(f"- {p}")

# Ask user to select the correct port
port_name = input("Enter COM port (e.g., COM3): ")

# Connect to the serial port
try:
    ser = serial.Serial(port_name, 115200, timeout=1)
    print(f"Connected to {port_name}")
except Exception as e:
    print(f"Error connecting to {port_name}: {e}")
    exit(1)

# Clear any initial data
ser.reset_input_buffer()
ser.reset_output_buffer()

# Give ESP32 time to initialize
print("Waiting for ESP32 to initialize...")
time.sleep(5)  # Increased wait time to 5 seconds

# Check if we're receiving data
print("Checking for data...")
start_check = time.time()
data_received = False
while time.time() - start_check < 5:  # Check for 5 seconds
    if ser.in_waiting > 0:
        print("Data detected!")
        data_received = True
        break
    time.sleep(0.1)

if not data_received:
    print("WARNING: No data detected from ESP32. Make sure your ESP32 is sending data.")
    proceed = input("Continue anyway? (y/n): ")
    if proceed.lower() != 'y':
        ser.close()
        exit(1)

# Function to parse the ESP32 data format
def parse_sensor_data(line):
    try:
        # Convert to proper JSON by adding quotes around keys
        json_str = re.sub(r'([a-z0-9]+):', r'"\1":', line)
        # Handle any missing closing brace
        if '}' not in json_str:
            json_str += '}'
        data = json.loads(json_str)
        
        # Extract values
        timestamp = data.get('t', 0)
        
        # Front sensor
        acc_x1 = data.get('ax1', 0)
        acc_y1 = data.get('ay1', 0)
        acc_z1 = data.get('az1', 0)
        gyr_x1 = data.get('gx1', 0)
        gyr_y1 = data.get('gy1', 0)
        gyr_z1 = data.get('gz1', 0)
        
        # Rear sensor
        acc_x2 = data.get('ax2', 0)
        acc_y2 = data.get('ay2', 0)
        acc_z2 = data.get('az2', 0)
        gyr_x2 = data.get('gx2', 0)
        gyr_y2 = data.get('gy2', 0)
        gyr_z2 = data.get('gz2', 0)
        
        # Default pothole marker (will be labeled in processing)
        is_pothole = 0
        
        return f"{timestamp},{acc_x1},{acc_y1},{acc_z1},{acc_x2},{acc_y2},{acc_z2},{gyr_x1},{gyr_y1},{gyr_z1},{gyr_x2},{gyr_y2},{gyr_z2},{is_pothole}"
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

# Start data collection
print(f"Starting {COLLECTION_PHASE} data collection for {COLLECTION_TIME} seconds...")


# Open in text mode with CSV header
with open(FILENAME, 'w') as f:
    # Write the CSV header
    f.write(CSV_HEADER + '\n')
    
    # Write for specified duration
    start_time = time.time()
    lines_written = 0
    
    # Add countdown
    print(f"Collection starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("GO! Ride Smooth!")
    
    while time.time() - start_time < COLLECTION_TIME:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='replace').rstrip()
            
            # Parse the JSON-like data
            csv_line = parse_sensor_data(line)
            if csv_line:
                print(csv_line)  # Display for monitoring
                f.write(csv_line + '\n')
                lines_written += 1
                
                # Flush occasionally
                if lines_written % 10 == 0:
                    f.flush()
        else:
            # Small sleep when no data
            time.sleep(0.01)
            
        # Show remaining time every 5 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
            print(f"Time remaining: {COLLECTION_TIME - int(elapsed)} seconds")

    print(f"Data collection complete! Wrote {lines_written} lines to {FILENAME}")

ser.close()