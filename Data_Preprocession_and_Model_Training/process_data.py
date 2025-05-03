import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import json
import re

def parse_json_like(data_file):
    with open(data_file, 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith('{"ax":'):
        print(f"Detected JSON-like format in {data_file}, converting...")
        
        parsed_lines = []
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    json_str = re.sub(r'([a-z]+):', r'"\1":', line.strip())
                    if '}' not in json_str:
                        json_str += '}'
                    data = json.loads(json_str)
                    
                    timestamp = data.get('t', 0)
                    acc_x1 = data.get('ax', 0)
                    acc_y1 = data.get('ay', 0)
                    acc_z1 = data.get('az', 0)
                    gyr_x1 = data.get('gx', 0)
                    gyr_y1 = data.get('gy', 0)
                    gyr_z1 = data.get('gz', 0)
                    
                    parsed_lines.append([timestamp, acc_x1, acc_y1, acc_z1, 
                                         acc_x1, acc_y1, acc_z1,
                                         gyr_x1, gyr_y1, gyr_z1,
                                         gyr_x1, gyr_y1, gyr_z1,
                                         0])
                except:
                    continue
                    
        columns = ["timestamp", "acc_x1", "acc_y1", "acc_z1", 
                  "acc_x2", "acc_y2", "acc_z2", 
                  "gyr_x1", "gyr_y1", "gyr_z1",
                  "gyr_x2", "gyr_y2", "gyr_z2",
                  "is_pothole"]
        return pd.DataFrame(parsed_lines, columns=columns)
    else:
        return pd.read_csv(data_file)

def find_data_files():
    data_files = []
    
    if os.path.exists('simulated_pothole_data.csv'):
        data_files.append(('simulated_pothole', 'simulated_pothole_data.csv'))
    
    if os.path.exists('baseline_data.csv'):
        data_files.append(('baseline', 'baseline_data.csv'))
    
    if os.path.exists('merged_internet_data.csv'):
        data_files.append(('internet', 'merged_internet_data.csv'))
    
    for phase in ['idle_engine', 'smooth_road', 'rough_road', 'pothole_road']:
        if os.path.exists(f'{phase}_data.csv'):
            data_files.append((phase, f'{phase}_data.csv'))
    
    return data_files

def identify_engine_noise(data, fs=100):
    idle_data = data.iloc[:min(1000, len(data))]
    
    engine_freqs = []
    
    for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
        f, Pxx = signal.periodogram(idle_data[axis], fs)
        
        peak_indices = signal.find_peaks(Pxx)[0]
        if len(peak_indices) > 0:
            sorted_peaks = peak_indices[np.argsort(Pxx[peak_indices])[-3:]]
            freqs = f[sorted_peaks]
            engine_freqs.extend(freqs)
            
            print(f"Engine frequencies in {axis}: {freqs} Hz")
            
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
    
    unique_freqs = np.unique(engine_freqs)
    return unique_freqs

def filter_engine_noise(data, engine_freqs, fs=100):
    filtered_data = data.copy()
    
    for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
        signal_data = data[axis].values
        
        for freq in engine_freqs:
            if freq > 0.5 and freq < 45:
                b, a = signal.iirnotch(freq, 30, fs)
                signal_data = signal.filtfilt(b, a, signal_data)
        
        b, h = signal.butter(3, 0.5, 'highpass', fs=fs)
        signal_data = signal.filtfilt(b, h, signal_data)
        
        filtered_data[axis] = signal_data
    
    # Calculate magnitudes
    filtered_data['acc_mag1'] = np.sqrt(filtered_data['acc_x1']**2 + 
                                     filtered_data['acc_y1']**2 + 
                                     filtered_data['acc_z1']**2)
    filtered_data['acc_mag2'] = np.sqrt(filtered_data['acc_x2']**2 + 
                                     filtered_data['acc_y2']**2 + 
                                     filtered_data['acc_z2']**2)
    
    # Calculate gyro magnitudes
    filtered_data['gyr_mag1'] = np.sqrt(filtered_data['gyr_x1']**2 + 
                                     filtered_data['gyr_y1']**2 + 
                                     filtered_data['gyr_z1']**2)
    filtered_data['gyr_mag2'] = np.sqrt(filtered_data['gyr_x2']**2 + 
                                     filtered_data['gyr_y2']**2 + 
                                     filtered_data['gyr_z2']**2)
    
    # Calculate jerk (derivative of acceleration)
    for axis in ['acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2']:
        filtered_data[f'{axis}_jerk'] = filtered_data[axis].diff() / (filtered_data['timestamp'].diff() / 1000)
    
    # Calculate sensor differences
    for axis in ['x', 'y', 'z']:
        filtered_data[f'acc_{axis}_diff'] = filtered_data[f'acc_{axis}1'] - filtered_data[f'acc_{axis}2']
        filtered_data[f'gyr_{axis}_diff'] = filtered_data[f'gyr_{axis}1'] - filtered_data[f'gyr_{axis}2']
    
    # Calculate z-score for key metrics
    for col in ['acc_z1', 'acc_z2', 'acc_mag1', 'acc_mag2']:
        rolling = filtered_data[col].rolling(window=20, center=True)
        filtered_data[f'{col}_zscore'] = (filtered_data[col] - rolling.mean()) / rolling.std().fillna(1)
    
    return filtered_data

def auto_label_potholes(data, threshold=1.8, window_size=8):
    data['auto_pothole'] = 0
    
    # Different thresholds for different sensors
    z1_threshold = -threshold * 0.9  # Slightly more sensitive for front sensor
    z2_threshold = -threshold * 1.1  # Slightly less sensitive for rear sensor
    mag_threshold = threshold + 3.5  # Reduced from +4 for better sensitivity
    jerk_threshold = 15.0  # Threshold for jerk detection
    zscore_threshold = 2.5  # Z-score threshold
    
    # Calculate ratio between vertical and horizontal accelerations
    data['z_xy_ratio1'] = np.abs(data['acc_z1']) / (np.abs(data['acc_x1']) + np.abs(data['acc_y1']) + 0.1)
    data['z_xy_ratio2'] = np.abs(data['acc_z2']) / (np.abs(data['acc_x2']) + np.abs(data['acc_y2']) + 0.1)
    ratio_threshold = 0.8  # Threshold for z/xy ratio
    
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i+window_size]
        
        # Z-axis condition with different thresholds for each sensor
        z_condition = (
            (window['acc_z1'].min() < z1_threshold) or 
            (window['acc_z2'].min() < z2_threshold)
        )
        
        # Magnitude condition
        mag_condition = (
            (window['acc_mag1'].max() > mag_threshold) or 
            (window['acc_mag2'].max() > mag_threshold)
        )
        
        # Jerk condition (if jerk columns exist)
        jerk_condition = False
        if 'acc_z1_jerk' in window.columns:
            jerk_condition = (
                (window['acc_z1_jerk'].abs().max() > jerk_threshold) or
                (window['acc_z2_jerk'].abs().max() > jerk_threshold)
            )
        
        # Z-score condition (if zscore columns exist)
        zscore_condition = False
        if 'acc_z1_zscore' in window.columns:
            zscore_condition = (
                (window['acc_z1_zscore'].abs().max() > zscore_threshold) or
                (window['acc_z2_zscore'].abs().max() > zscore_threshold)
            )
        
        # Ratio condition
        ratio_condition = (
            (window['z_xy_ratio1'].max() > ratio_threshold) or
            (window['z_xy_ratio2'].max() > ratio_threshold)
        )
        
        # Duration condition - require at least 3 consecutive samples above threshold
        duration_condition = (
            (window['acc_z1'] < z1_threshold).sum() >= 3 or
            (window['acc_z2'] < z2_threshold).sum() >= 3
        )
        
        # Combined condition - require at least 3 of the 5 conditions to be true
        conditions_met = sum([
            z_condition, 
            mag_condition, 
            jerk_condition, 
            zscore_condition,
            ratio_condition,
            duration_condition
        ])
        
        if conditions_met >= 3:
            data.loc[data.index[i:i+window_size], 'auto_pothole'] = 1
    
    # Group consecutive pothole detections
    pothole_groups = []
    in_pothole = False
    start_idx = 0
    
    for idx, is_pothole in enumerate(data['auto_pothole']):
        if is_pothole and not in_pothole:
            in_pothole = True
            start_idx = idx
        elif not is_pothole and in_pothole:
            in_pothole = False
            pothole_groups.append((start_idx, idx))
    
    if in_pothole:
        pothole_groups.append((start_idx, len(data['auto_pothole'])))
    
    print(f"Automatically detected {len(pothole_groups)} potential potholes")
    return data, pothole_groups

def create_visualizations(data, filtered_data, labeled_data, pothole_groups, phase):
    plt.figure(figsize=(15, 10))
    
    sample_window = min(2000, len(data))
    
    plt.subplot(2, 1, 1)
    plt.plot(data['timestamp'][:sample_window]/1000, data['acc_z1'][:sample_window], label='Front Z (Raw)')
    plt.plot(data['timestamp'][:sample_window]/1000, data['acc_z2'][:sample_window], label='Rear Z (Raw)')
    plt.title(f'Raw Acceleration Data - {phase.upper()}')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    
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
    
    plt.figure(figsize=(15, 10))
    
    sample_window = min(5000, len(labeled_data))
    
    plt.plot(labeled_data['timestamp'][:sample_window]/1000, labeled_data['acc_z1'][:sample_window], label='Front Z')
    plt.plot(labeled_data['timestamp'][:sample_window]/1000, labeled_data['acc_z2'][:sample_window], label='Rear Z')
    
    for start, end in pothole_groups:
        if start < sample_window:
            end_idx = min(end, sample_window-1)  # Ensure end_idx is within bounds
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

def process_data():
    data_files = find_data_files()
    if not data_files:
        print("No data files found! Please collect data first.")
        return
    
    print(f"Found {len(data_files)} data files to process.")
    
    for phase, file_path in data_files:
        print(f"\nProcessing {phase} data from {file_path}...")
        
        data = parse_json_like(file_path)
        print(f"Loaded {len(data)} samples")
        
        required_columns = ["acc_x1", "acc_y1", "acc_z1", "acc_x2", "acc_y2", "acc_z2"]
        if not all(col in data.columns for col in required_columns):
            print(f"WARNING: Missing required columns in {file_path}!")
            print(f"Available columns: {data.columns.tolist()}")
            continue
        
        data['acc_mag1'] = np.sqrt(data['acc_x1']**2 + data['acc_y1']**2 + data['acc_z1']**2)
        data['acc_mag2'] = np.sqrt(data['acc_x2']**2 + data['acc_y2']**2 + data['acc_z2']**2)
        
        if phase == 'idle_engine':
            engine_freqs = identify_engine_noise(data)
            np.save('engine_frequencies.npy', engine_freqs)
        else:
            if os.path.exists('engine_frequencies.npy'):
                engine_freqs = np.load('engine_frequencies.npy')
                print(f"Loaded {len(engine_freqs)} engine frequencies")
            else:
                engine_freqs = []
        
        if len(engine_freqs) > 0:
            filtered_data = filter_engine_noise(data, engine_freqs)
        else:
            filtered_data = data
            
        # Refined thresholds for different road conditions
        thresholds = {
            'default': 1.8,
            'simulated_pothole': 1.3,
            'pothole_road': 2.0,  # Reduced from 2.2 for better sensitivity
            'rough_road': 2.8,    # Increased from 2.5 to reduce false positives
            'smooth_road': 1.5,
            'idle_engine': 3.0    # High threshold to avoid false positives during idle
        }
        
        threshold = thresholds.get(phase, thresholds['default'])
        
        labeled_data, pothole_groups = auto_label_potholes(filtered_data, threshold)
        
        output_file = f'processed_{phase}_data.csv'
        labeled_data.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        
        create_visualizations(data, filtered_data, labeled_data, pothole_groups, phase)
    
    print("\nAll data processing complete!")

if __name__ == "__main__":
    process_data()
