import cv2
import numpy as np
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("SOS")
    exit()

def non_max_suppression_fast(image, kernel_size=3):
    """Apply Non-Maximum Suppression (NMS) to enhance edges."""
    dilated = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))
    nms_result = np.where(image == dilated, image, 0)
    return nms_result

# Create or open CSV file with headers
csv_file = 'pupil_detection_log.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Detection Status', 'Number of Circles'])

# Lists to store detection results
timestamps = []
detection_status = []  # 1 for detection, 0 for miss

while True:
    start_time = time.time()  # Start time for FPS calculation

    ret, frame = cap.read()
    if not ret:
        print("Fail")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    fg_mask = cv2.adaptiveThreshold(fg_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    smooth_gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(smooth_gray, 80, 180)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    edges_nms = non_max_suppression_fast(edges)
    edges_nms_colored = cv2.cvtColor(edges_nms, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(edges_nms, cv2.HOUGH_GRADIENT, dp=2.0, minDist=50, 
                             param1=10, param2=60, minRadius=10, maxRadius=20)

    timestamp = datetime.now()
    timestamps.append(timestamp)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        num_circles = len(circles[0])
        detection_status.append(1)  # Success
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 'Detected', num_circles])
        
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center point
    else:
        detection_status.append(0)  # Miss
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 'Missed', 0])

    # FPS calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize images
    width, height = 1600, 900
    frame_resized = cv2.resize(frame, (width, height))
    fg_resized = cv2.resize(fg_colored, (width, height))
    edges_resized = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (width, height))

    # Stack images in 2x2 grid
    top_row = np.hstack((frame_resized, fg_resized))
    bottom_row = np.hstack((edges_resized, frame_resized))
    combined_view = np.vstack((top_row, bottom_row))

    # Show the final output
    cv2.imshow("Original | Background Removed | Canny | Hough Circles (NMS)", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
    plt.savefig('pupil_detection_success_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'pupil_detection_success_plot.png'")
    print(f"Final success rate: {final_success_rate:.1f}%")
else:
    print("No frames processed for plotting")

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()