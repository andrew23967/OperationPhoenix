import cv2
from bot_recognition import get_orientation
from crop import rescale
import numpy as np

cap = cv2.VideoCapture(0)

center = (0, 0)
angle = 0

while 1:
    ret, frame = cap.read()

    frame = rescale(frame)

    try:
        center, angle = get_orientation(frame)
    except:
        pass

    center = (int(center[0]), int(center[1]))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    p2 = (int(center[0] + 50 * np.cos(angle)), int(center[1] + 50 * np.sin(angle)))

    cv2.line(frame, center, p2, (255, 0, 0), 1)

    cv2.imshow('test_video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
