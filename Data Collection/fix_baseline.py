import pandas as pd

# Load baseline_data.csv
baseline_data_path = 'baseline_data.csv'
baseline_data = pd.read_csv(baseline_data_path)

# Define the correct headers
correct_headers = ['timestamp', 'acc_x1', 'acc_y1', 'acc_z1', 'acc_x2', 'acc_y2', 'acc_z2', 
                  'gyr_x1', 'gyr_y1', 'gyr_z1', 'gyr_x2', 'gyr_y2', 'gyr_z2', 'is_pothole']

# Rename the columns
baseline_data.columns = correct_headers

# Save the corrected file
baseline_data.to_csv('baseline_data.csv', index=False)

print("Baseline data headers fixed successfully!")
