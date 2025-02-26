import cv2
import numpy as np
import os

CHESSBOARD_SIZE = (8, 5)
FRAME_COUNT = 10
SAVE_FOLDER = "calibration_images"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(1)
frame_count = 0

while frame_count < FRAME_COUNT:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)

        # Save frame
        image_path = os.path.join(SAVE_FOLDER, f"frame_{frame_count+1}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

        frame_count += 1

    progress_text = f"Calibration Progress: {frame_count}/{FRAME_COUNT}"
    cv2.putText(frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
print("\n=== Camera Calibration Results ===")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", distortion_coeffs)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)
