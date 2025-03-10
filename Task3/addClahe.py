import cv2
import numpy as np
import time
import csv
from datetime import datetime

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
csv_file = 'pupil_detection_log_v3.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Detection Status', 'Number of Circles'])

# Initialize CLAHE for better contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    start_time = time.time()  # Start time for FPS calculation

    ret, frame = cap.read()
    if not ret:
        print("Fail")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE instead of regular histogram equalization
    gray = clahe.apply(gray)

    # Stronger bilateral filter for noise reduction
    smooth_gray = cv2.bilateralFilter(gray, 11, 100, 100)  # Increased kernel size and sigma
    
    # Background subtraction
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    fg_mask = cv2.adaptiveThreshold(fg_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # Optimized Canny edge detection
    edges = cv2.Canny(smooth_gray, 50, 150)  # Lowered thresholds for more edges
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  # Connect broken edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    edges_nms = non_max_suppression_fast(edges)
    edges_nms_colored = cv2.cvtColor(edges_nms, cv2.COLOR_GRAY2BGR)

    # Optimized Hough Circle Transform
    circles = cv2.HoughCircles(
        edges_nms, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,           # Lowered for higher resolution
        minDist=40,       # Adjusted based on typical eye distance
        param1=50,        # Lowered to match Canny upper threshold
        param2=30,        # Lowered for higher sensitivity
        minRadius=8,      # Adjusted range for pupil size
        maxRadius=25
    )

    timestamp = datetime.now()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        num_circles = len(circles[0])
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 'Detected', num_circles])
        
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center point
    else:
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

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Detection data saved to {csv_file}. Run 'plot_results.py' to generate the plot.")