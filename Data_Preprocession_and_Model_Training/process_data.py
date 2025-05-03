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
    
    filtered_data['acc_mag1'] = np.sqrt(filtered_data['acc_x1']**2 + 
                                     filtered_data['acc_y1']**2 + 
                                     filtered_data['acc_z1']**2)
    filtered_data['acc_mag2'] = np.sqrt(filtered_data['acc_x2']**2 + 
                                     filtered_data['acc_y2']**2 + 
                                     filtered_data['acc_z2']**2)
    
    return filtered_data

def auto_label_potholes(data, threshold=1.8, window_size=5):
    data['auto_pothole'] = 0
    
    # Modified pothole detection condition with lower magnitude threshold
    z_threshold = -threshold
    mag_threshold = threshold + 4  # Reduced from +6 to +4 for better sensitivity
    
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i+window_size]
        
        z_condition = (
            (window['acc_z1'].min() < z_threshold) or 
            (window['acc_z2'].min() < z_threshold)
        )
        
        mag_condition = (
            (window['acc_mag1'].max() > mag_threshold) or 
            (window['acc_mag2'].max() > mag_threshold)
        )
        
        if z_condition and mag_condition:
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
            end_idx = min(end, sample_window)
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
            'pothole_road': 2.2,
            'rough_road': 2.5,
            'smooth_road': 1.5,
            'idle_engine': 3.0  # High threshold to avoid false positives during idle
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
