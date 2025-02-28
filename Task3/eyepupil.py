import cv2
import numpy as np
import time

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("SOS")
    exit()

def non_max_suppression_fast(image, kernel_size=3):
    """Apply Non-Maximum Suppression (NMS) to enhance edges."""
    dilated = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))
    nms_result = np.where(image == dilated, image, 0)
    return nms_result

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

    circles = cv2.HoughCircles(edges_nms, cv2.HOUGH_GRADIENT, dp=2.0, minDist=50, param1=10, param2=60, minRadius=10, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center point


    # FPS calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Resize images
    width, height = 1600, 900
    frame_resized = cv2.resize(frame, (width, height))
    fg_resized = cv2.resize(fg_colored, (width, height))
    edges_resized = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (width, height))

    # Stack images in 2x2 grid
    top_row = np.hstack((frame_resized, fg_resized))  # First row: Original (with circles) & Background Removed
    bottom_row = np.hstack((edges_resized, frame_resized))  # Second row: Canny & Real Frame (with circles)
    combined_view = np.vstack((top_row, bottom_row))  # Final 2x2 grid

    # Show the final output
    cv2.imshow("Original | Background Removed | Canny | Hough Circles (NMS)", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
