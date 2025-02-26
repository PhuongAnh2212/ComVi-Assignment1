import cv2
import numpy as np
import time

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("SOS")
    exit()

while True:
    start_time = time.time()  # Start time for FPS calculation

    ret, frame = cap.read()

    if not ret: 
        print("Fail")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=70, param2=45, minRadius=10, maxRadius=15)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)

    fps = 1.0 / (time.time() - start_time)

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pupil Tracking', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()