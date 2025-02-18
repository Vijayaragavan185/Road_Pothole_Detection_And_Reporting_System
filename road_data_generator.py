import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

class RoadDataGenerator:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
    def generate_normal_road(self, duration):
        """Generate data for normal road conditions"""
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Base vibration
        acc_x = 0.2 * np.random.randn(samples)
        acc_y = 0.2 * np.random.randn(samples)
        acc_z = 0.5 + 0.3 * np.random.randn(samples)  # Vertical acceleration
        
        # Add some road noise
        acc_z += 0.2 * np.sin(2 * np.pi * 5 * t)
        
        return acc_x, acc_y, acc_z, np.zeros(samples)  # Last array is for labels
    
    def generate_pothole(self, duration=0.5):
        """Generate data for pothole impact"""
        samples = int(duration * self.sampling_rate)
        
        # Create time array for the gaussian pulse
        t = np.linspace(-3, 3, samples)  # Normalized time array
        
        # Generate gaussian pulse
        pulse = np.exp(-(t**2))
        
        # Scale the pulses for each axis
        acc_x = 0.5 * pulse + 0.1 * np.random.randn(samples)
        acc_y = 0.3 * pulse + 0.1 * np.random.randn(samples)
        acc_z = -2.0 * pulse + 0.1 * np.random.randn(samples)  # Stronger vertical impact
        
        return acc_x, acc_y, acc_z, np.ones(samples)  # Last array is for labels
    
    def generate_dataset(self, normal_duration=60, num_potholes=20):
        """Generate complete dataset with normal roads and potholes"""
        # Generate normal road data
        normal_x, normal_y, normal_z, normal_labels = self.generate_normal_road(normal_duration)
        
        # Insert random potholes
        for _ in range(num_potholes):
            # Random position for pothole
            pos = np.random.randint(0, len(normal_x) - 50)
            
            # Generate pothole data
            p_x, p_y, p_z, p_label = self.generate_pothole()
            
            # Insert pothole data
            slice_length = len(p_x)
            normal_x[pos:pos+slice_length] = p_x
            normal_y[pos:pos+slice_length] = p_y
            normal_z[pos:pos+slice_length] = p_z
            normal_labels[pos:pos+slice_length] = p_label
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': np.arange(len(normal_x)) / self.sampling_rate,
            'acc_x': normal_x,
            'acc_y': normal_y,
            'acc_z': normal_z,
            'label': normal_labels
        })
        
        return data
    
    def plot_data(self, data, start_idx=0, duration=5):
        """Plot a section of the generated data"""
        samples = int(duration * self.sampling_rate)
        end_idx = start_idx + samples
        
        plt.figure(figsize=(15, 10))
        
        # Plot accelerometer data
        plt.subplot(4, 1, 1)
        plt.plot(data['timestamp'][start_idx:end_idx], data['acc_x'][start_idx:end_idx])
        plt.title('X-axis Acceleration')
        plt.ylabel('g')
        
        plt.subplot(4, 1, 2)
        plt.plot(data['timestamp'][start_idx:end_idx], data['acc_y'][start_idx:end_idx])
        plt.title('Y-axis Acceleration')
        plt.ylabel('g')
        
        plt.subplot(4, 1, 3)
        plt.plot(data['timestamp'][start_idx:end_idx], data['acc_z'][start_idx:end_idx])
        plt.title('Z-axis Acceleration')
        plt.ylabel('g')
        
        # Plot labels
        plt.subplot(4, 1, 4)
        plt.plot(data['timestamp'][start_idx:end_idx], data['label'][start_idx:end_idx])
        plt.title('Labels (1 = Pothole)')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
   
    generator = RoadDataGenerator()
    
   
    print("Generating dataset...")
    data = generator.generate_dataset(normal_duration=60, num_potholes=20)
    
    # Plot first 5 seconds of data
    print("Plotting data...")
    generator.plot_data(data, duration=5)
    
    
    print("Saving dataset...")
    data.to_csv('synthetic_road_data.csv', index=False)
    print(f"Generated dataset with {len(data)} samples and {int(data['label'].sum())} pothole instances")