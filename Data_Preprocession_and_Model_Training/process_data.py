import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import json
import re

# Helper function to parse JSON-like data if needed
def parse_json_like(data_file):
    """Detect and parse JSON-like data from ESP32"""
    with open(data_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if it's our JSON format (support both single and dual sensor formats)
    if first_line.startswith('{"ax1":') or first_line.startswith('{"ax":'):
        print(f"Detected JSON-like format in {data_file}, converting...")
        
        parsed_lines = []
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    # Convert to proper JSON
                    json_str = re.sub(r'([a-z0-9]+):', r'"\1":', line.strip())
                    if '}' not in json_str:
                        json_str += '}'
                    data = json.loads(json_str)
                    
                    # Extract values - support both formats
                    timestamp = data.get('t', 0)
                    
                    # Try dual sensor format first
                    if 'ax1' in data:
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
                    else:
                        # Single sensor format (duplicate as both sensors)
                        acc_x1 = data.get('ax', 0)
                        acc_y1 = data.get('ay', 0)
                        acc_z1 = data.get('az', 0)
                        gyr_x1 = data.get('gx', 0)
                        gyr_y1 = data.get('gy', 0)
                        gyr_z1 = data.get('gz', 0)
                        
                        # Duplicate as sensor 2
                        acc_x2 = acc_x1
                        acc_y2 = acc_y1
                        acc_z2 = acc_z1
                        gyr_x2 = gyr_x1
                        gyr_y2 = gyr_y1
                        gyr_z2 = gyr_z1
                    
                    parsed_lines.append([timestamp, acc_x1, acc_y1, acc_z1, 
                                       acc_x2, acc_y2, acc_z2,
                                       gyr_x1, gyr_y1, gyr_z1,
                                       gyr_x2, gyr_y2, gyr_z2,
                                       0])  # is_pothole
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
                    
        # Create DataFrame
        columns = ["timestamp", "acc_x1", "acc_y1", "acc_z1", 
                  "acc_x2", "acc_y2", "acc_z2", 
                  "gyr_x1", "gyr_y1", "gyr_z1",
                  "gyr_x2", "gyr_y2", "gyr_z2",
                  "is_pothole"]
        return pd.DataFrame(parsed_lines, columns=columns)
    else:
        # Regular CSV file
        return pd.read_csv(data_file)

# Function to find all data files
def find_data_files():
    """Find all data files in current directory"""
    data_files = []
    
    # Look for simulated pothole data
    if os.path.exists('simulated_pothole_data.csv'):
        data_files.append(('simulated_pothole', 'simulated_pothole_data.csv'))
    
    # Look for baseline data
    if os.path.exists('baseline_data.csv'):
        data_files.append(('baseline', 'baseline_data.csv'))
    
    # Look for internet data
    if os.path.exists('merged_internet_data.csv'):
        data_files.append(('internet', 'merged_internet_data.csv'))
    
    # Look for motorcycle data
    for phase in ['idle_engine', 'smooth_road', 'rough_road', 'pothole_road']:
        if os.path.exists(f'{phase}_data.csv'):
            data_files.append((phase, f'{phase}_data.csv'))
    
    return data_files

# Identify engine noise frequencies
def identify_engine_noise(data, fs=100):
    """Identify dominant engine noise frequencies"""
    # Use first 1000 samples (assuming motorcycle was idling)
    idle_data = data.iloc[:min(1000, len(data))]
    
    engine_freqs = []
    
    # Analyze frequency content for each axis
    for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
        f, Pxx = signal.periodogram(idle_data[axis], fs)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(Pxx)[0]
        if len(peak_indices) > 0:
            sorted_peaks = peak_indices[np.argsort(Pxx[peak_indices])[-3:]]
            freqs = f[sorted_peaks]
            engine_freqs.extend(freqs)
            
            print(f"Engine frequencies in {axis}: {freqs} Hz")
            
            # Plot
            plt.figure(figsize=(10, 4))
            plt.semilogy(f, Pxx)
            plt.title(f'Engine Noise Spectrum - {axis}')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power Spectral Density')
            for freq in freqs:
                plt.axvline(x=freq, color='r', linestyle='--')
            plt.grid()
            plt.savefig(f'engine_noise_{axis}.png')
            plt.close()
    
    # Get unique frequencies
    unique_freqs = np.unique(engine_freqs)
    return unique_freqs

# Filter out engine noise
def filter_engine_noise(data, engine_freqs, fs=100):
    """Filter out engine noise frequencies"""
    filtered_data = data.copy()
    
    for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
        signal_data = data[axis].values
        
        # Apply notch filters for engine frequencies
        for freq in engine_freqs:
            if freq > 0.5 and freq < 45:  # Only filter reasonable frequencies
                b, a = signal.iirnotch(freq, 30, fs)
                signal_data = signal.filtfilt(b, a, signal_data)
        
        # High-pass filter to remove very low frequency drift
        b, h = signal.butter(3, 0.5, 'highpass', fs=fs)
        signal_data = signal.filtfilt(b, h, signal_data)
        
        filtered_data[axis] = signal_data
    
    # Recalculate magnitudes
    filtered_data['acc_mag1'] = np.sqrt(filtered_data['acc_x1']**2 + 
                                     filtered_data['acc_y1']**2 + 
                                     filtered_data['acc_z1']**2)
    filtered_data['acc_mag2'] = np.sqrt(filtered_data['acc_x2']**2 + 
                                     filtered_data['acc_y2']**2 + 
                                     filtered_data['acc_z2']**2)
    
    return filtered_data

# Auto-label potential potholes

def auto_label_potholes(data, threshold=2.0):
    """Auto-label potential potholes using acceleration thresholds"""
    # Add auto_pothole column
    data['auto_pothole'] = 0
    
    # Mark as pothole if vertical acceleration or magnitude exceeds threshold
    pothole_condition = (
        (data['acc_z1'] < -threshold) | 
        (data['acc_z2'] < -threshold) | 
        (data['acc_mag1'] > threshold+6) | 
        (data['acc_mag2'] > threshold+6)
    )
    
    data.loc[pothole_condition, 'auto_pothole'] = 1
    
    # Group consecutive pothole detections
    pothole_groups = []
    in_pothole = False
    start_idx = 0
    
    for idx, is_pothole in enumerate(pothole_condition):
        if is_pothole and not in_pothole:
            in_pothole = True
            start_idx = idx
        elif not is_pothole and in_pothole:
            in_pothole = False
            pothole_groups.append((start_idx, idx))
    
    # Add last group if ending in pothole
    if in_pothole:
        pothole_groups.append((start_idx, len(pothole_condition) - 1))
    
    print(f"Automatically detected {len(pothole_groups)} potential potholes")
    return data, pothole_groups

# Create visualizations of the processed data
def create_visualizations(data, filtered_data, labeled_data, pothole_groups, phase):
    """Create visualizations of the data processing steps"""
    # 1. Raw vs Filtered Data
    plt.figure(figsize=(15, 10))
    
    # Limit to 2000 samples for visualization
    sample_window = min(2000, len(data))
    
    # Before filtering
    plt.subplot(2, 1, 1)
    plt.plot(data['timestamp'][:sample_window]/1000, data['acc_z1'][:sample_window], label='Front Z (Raw)')
    plt.plot(data['timestamp'][:sample_window]/1000, data['acc_z2'][:sample_window], label='Rear Z (Raw)')
    plt.title(f'Raw Acceleration Data - {phase.upper()}')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    
    # After filtering
    plt.subplot(2, 1, 2)
    plt.plot(filtered_data['timestamp'][:sample_window]/1000, filtered_data['acc_z1'][:sample_window], label='Front Z (Filtered)')
    plt.plot(filtered_data['timestamp'][:sample_window]/1000, filtered_data['acc_z2'][:sample_window], label='Rear Z (Filtered)')
    plt.title('Filtered Acceleration Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'filtering_comparison_{phase}.png')
    plt.close()
    
    # 2. Pothole Detection Results
    plt.figure(figsize=(15, 10))
    
    # Show more data for pothole detection
    sample_window = min(5000, len(labeled_data))
    
    plt.plot(labeled_data['timestamp'][:sample_window]/1000, labeled_data['acc_z1'][:sample_window], label='Front Z')
    plt.plot(labeled_data['timestamp'][:sample_window]/1000, labeled_data['acc_z2'][:sample_window], label='Rear Z')
    
    # Highlight pothole regions
    for start, end in pothole_groups:
        if start < sample_window:
            end_idx = min(end, sample_window)
            # Ensure end_idx does not exceed the DataFrame's maximum index
            end_idx = min(end_idx, len(labeled_data) - 1)
            plt.axvspan(labeled_data['timestamp'][start]/1000,
                    labeled_data['timestamp'][end_idx]/1000,
                    alpha=0.3, color='red')
    
    plt.title(f'Detected Potholes - {phase.upper()}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'pothole_detection_{phase}.png')
    plt.close()

# Main processing function
def process_data():
    data_files = find_data_files()
    if not data_files:
        print("No data files found! Please collect data first.")
        return
    
    print(f"Found {len(data_files)} data files to process.")
    
    # Process each file
    for phase, file_path in data_files:
        print(f"\nProcessing {phase} data from {file_path}...")
        
        # Load data (handle JSON-like format if needed)
        data = parse_json_like(file_path)
        print(f"Loaded {len(data)} samples")
        
        # Ensure required columns exist
        required_columns = ["acc_x1", "acc_y1", "acc_z1", "acc_x2", "acc_y2", "acc_z2"]
        if not all(col in data.columns for col in required_columns):
            print(f"WARNING: Missing required columns in {file_path}!")
            print(f"Available columns: {data.columns.tolist()}")
            continue
        
        # Calculate magnitudes
        data['acc_mag1'] = np.sqrt(data['acc_x1']**2 + data['acc_y1']**2 + data['acc_z1']**2)
        data['acc_mag2'] = np.sqrt(data['acc_x2']**2 + data['acc_y2']**2 + data['acc_z2']**2)
        
        # If this is idle engine data, use it to identify engine frequencies
        if phase == 'idle_engine':
            engine_freqs = identify_engine_noise(data)
            
            # Save engine frequencies for other files
            np.save('engine_frequencies.npy', engine_freqs)
        else:
            # Load engine frequencies if available
            if os.path.exists('engine_frequencies.npy'):
                engine_freqs = np.load('engine_frequencies.npy')
                print(f"Loaded {len(engine_freqs)} engine frequencies")
            else:
                engine_freqs = []
        
        # Filter engine noise if frequencies are available
        if len(engine_freqs) > 0:
            filtered_data = filter_engine_noise(data, engine_freqs)
        else:
            filtered_data = data
            
        # Auto-label potential potholes
        # Adjust threshold based on phase
        threshold = 2.0  # Default
        if phase == 'simulated_pothole':
            threshold = 1.5  # More sensitive for simulated potholes
        elif phase == 'pothole_road':
            threshold = 2.5  # More forgiving for real potholes
        
        labeled_data, pothole_groups = auto_label_potholes(filtered_data, threshold)
        
        # Save processed data
        output_file = f'processed_{phase}_data.csv'
        labeled_data.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        
        # Create visualizations
        create_visualizations(data, filtered_data, labeled_data, pothole_groups, phase)
    
    print("\nAll data processing complete!")

if __name__ == "__main__":
    process_data()