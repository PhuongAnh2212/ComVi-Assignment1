import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read data from CSV
csv_file = 'pupil_detection_log_2.csv'
timestamps = []
detection_status = []

try:
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            timestamp = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
            status = 1 if row[1] == 'Detected' else 0
            timestamps.append(timestamp)
            detection_status.append(status)
except FileNotFoundError:
    print(f"Error: {csv_file} not found. Please run pupil_detection.py first.")
    exit()

# Calculate cumulative success percentage and plot
if timestamps:  # Only create plot if there were frames processed
    # Calculate running success percentage
    cumulative_detections = np.cumsum(detection_status)
    total_frames = np.arange(1, len(detection_status) + 1)
    success_percentage = (cumulative_detections / total_frames) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, success_percentage, 'b-', label='Success Rate')
    plt.title('Pupil Detection Success Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Success Percentage (%)')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Set y-axis limits
    plt.ylim(0, 100)
    
    # Add final success rate text
    final_success_rate = success_percentage[-1]
    plt.text(timestamps[-1], final_success_rate, f'Final: {final_success_rate:.1f}%', 
             verticalalignment='bottom', horizontalalignment='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pupil_detection_log_2.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as pupil_detection_log_2.png'")
    print(f"Final success rate: {final_success_rate:.1f}%")
else:
    print("No data found in CSV for plotting")

plt.close()