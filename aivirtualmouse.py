import cv2
import numpy as np
import time
import HandTrackingModule as htm  # Ensure this module exists
import pyautogui

# Set camera dimensions
wCam, hCam = 640, 480
frameR = 100  # Frame reduction for smoother movement
smoothening = 4  # Smoothening factor for cursor movement

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)  # Ensure this class is correctly implemented

# Get screen size dynamically
wScr, hScr = pyautogui.size()

while True:
    # 1. Capture video frame
    success, img = cap.read()
    if not success:
        continue  # Skip this frame if the camera fails

    # 2. Detect hands
    img = detector.findHands(img)

    # 3. Find hand landmarks
    lmList, bbox = detector.findPosition(img)

    if lmList and len(lmList) > 12:  # Ensure landmarks exist
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        fingers = detector.fingersUp()

        # Draw rectangle around movement area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Move Mouse: Only Index Finger Up
        if fingers[1] == 1 and fingers[2] == 0:
            # Map coordinates from camera to screen
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse cursor
            pyautogui.moveTo(wScr - clocX, clocY, duration=0.05)

            # Draw a circle at fingertip
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # Update previous location
            plocX, plocY = clocX, clocY

        # Click Detection: Index & Middle Finger Up
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, _ = detector.findDistance(8, 12, img)
            if length < 40:  # If fingers are close enough
                pyautogui.click()
                time.sleep(0.1)  # Delay to prevent multiple clicks

    # 4. Compute FPS
    cTime = time.time()
    fps = 1 / max((cTime - pTime), 1e-5)  # Prevent division by zero
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 5. Display output
    cv2.imshow('Image', img)

    # 6. Allow exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()