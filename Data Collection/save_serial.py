import serial
import time
import serial.tools.list_ports

# First, let's find available COM ports
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

# Open file for writing
print("Starting data collection...")
with open('baseline_data.csv', 'wb') as f:
    # Write for 30 seconds
    start_time = time.time()
    bytes_written = 0
    while time.time() - start_time < 30:
        if ser.in_waiting > 0:
            line = ser.readline()
            # Try to decode and print for monitoring
            try:
                decoded = line.decode('utf-8', errors='replace').rstrip()
                print(decoded)
            except:
                print("Binary data received")
            
            # Write raw binary data to file
            f.write(line)
            bytes_written += len(line)
            
            # Flush the file every few lines to ensure data is written
            if bytes_written % 1000 < len(line):
                f.flush()
        else:
            # Small sleep to prevent CPU hogging when no data
            time.sleep(0.01)

    print(f"Data collection complete! Wrote {bytes_written} bytes.")

ser.close()