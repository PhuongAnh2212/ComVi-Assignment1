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

def is_pupil_candidate(image, center, radius, intensity_threshold=100):
    """Check if a detected circle is dark enough to be a pupil."""
    x, y = center
    r = radius
    # Define a small ROI around the circle
    roi = image[max(0, y-r):min(image.shape[0], y+r), max(0, x-r):min(image.shape[1], x+r)]
    if roi.size == 0:
        return False
    mean_intensity = np.mean(roi)
    return mean_intensity < intensity_threshold  # Pupils are typically dark

# Create or open CSV file with headers
csv_file = 'addRefine.csv'
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
    gray = clahe.apply(gray)

    # Stronger noise reduction
    smooth_gray = cv2.GaussianBlur(gray, (7, 7), 1.5)  # Increased blur to reduce noise
    
    # Background subtraction (for display)
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    fg_mask = cv2.adaptiveThreshold(fg_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # Edge detection with tuned thresholds
    edges = cv2.Canny(smooth_gray, 50, 150)
    # Morphological closing to connect pupil edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges_nms = non_max_suppression_fast(edges, kernel_size=5)  # Larger kernel for finer control
    edges_colored = cv2.cvtColor(edges_nms, cv2.COLOR_GRAY2BGR)

    # Hough Circle Transform with tighter parameters
    circles = cv2.HoughCircles(
        edges_nms, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,            # Slightly higher resolution
        minDist=50,        # Distance between circles
        param1=50,         # Matches Canny upper threshold
        param2=40,         # Increased to reduce false positives
        minRadius=15,      # Narrowed range for pupil size
        maxRadius=25
    )

    timestamp = datetime.now()
    circles_detected = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        valid_circles = []
        
        # Filter circles based on intensity
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            if is_pupil_candidate(gray, center, radius, intensity_threshold=100):
                valid_circles.append(i)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), 3)
        
        circles_detected = len(valid_circles)
    
    # Log detection result
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if circles_detected > 0:
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 'Detected', circles_detected])
        else:
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), 'Missed', 0])

    # FPS calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize images
    width, height = 1600, 900
    frame_resized = cv2.resize(frame, (width, height))
    fg_resized = cv2.resize(fg_colored, (width, height))
    edges_resized = cv2.resize(cv2.cvtColor(edges_nms, cv2.COLOR_GRAY2BGR), (width, height))

    # Stack images in 2x2 grid
    top_row = np.hstack((frame_resized, fg_resized))
    bottom_row = np.hstack((edges_resized, frame_resized))
    combined_view = np.vstack((top_row, bottom_row))

    # Show the final output
    cv2.imshow("Original | Background Removed | Canny | Hough Circles (NMS)", combined_view)

    # Optional: Show edges for debugging
    cv2.imshow("Edges NMS", edges_nms)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Detection data saved to {csv_file}. Run 'plot_results.py' to generate the plot.")