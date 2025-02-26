import cv2
import numpy as np

CHESSBOARD_SIZE = (8, 5)

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load image
image_path = "IMG_3754.jpg"  # Change this to your image file path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

    cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
    cv2.imshow("Chessboard", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# If no corners found, exit
if not objpoints or not imgpoints:
    print("Error: No corners found.")
    exit()

# Perform calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("\n=== Camera Calibration Results ===")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", distortion_coeffs)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)
