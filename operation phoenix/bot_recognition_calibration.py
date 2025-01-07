import cv2
import numpy as np
from crop import rescale

def nothing(x):
    pass

cv2.namedWindow('color_slider')

cv2.resizeWindow('color_slider', 400, 230)

cv2.createTrackbar('HMin', 'color_slider', 0, 255, nothing)
cv2.createTrackbar('SMin', 'color_slider', 0, 255, nothing)
cv2.createTrackbar('VMin', 'color_slider', 0, 255, nothing)
cv2.createTrackbar('HMax', 'color_slider', 0, 255, nothing)
cv2.createTrackbar('SMax', 'color_slider', 0, 255, nothing)
cv2.createTrackbar('VMax', 'color_slider', 0, 255, nothing)

hMin, sMin, vMin = (0, 0, 0)
hMax, sMax, vMax = (255, 255, 255)

cv2.setTrackbarPos('HMin', 'color_slider', hMin)
cv2.setTrackbarPos('SMin', 'color_slider', sMin)
cv2.setTrackbarPos('VMin', 'color_slider', vMin)
cv2.setTrackbarPos('HMax', 'color_slider', hMax)
cv2.setTrackbarPos('SMax', 'color_slider', sMax)
cv2.setTrackbarPos('VMax', 'color_slider', vMax)

cap = cv2.VideoCapture(0)

while 1:
    hMin = cv2.getTrackbarPos('HMin', 'color_slider')
    sMin = cv2.getTrackbarPos('SMin', 'color_slider')
    vMin = cv2.getTrackbarPos('VMin', 'color_slider')
    hMax = cv2.getTrackbarPos('HMax', 'color_slider')
    sMax = cv2.getTrackbarPos('SMax', 'color_slider')
    vMax = cv2.getTrackbarPos('VMax', 'color_slider')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    ret, frame = cap.read()

    frame = rescale(frame)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(frame_hsv, lower, upper)    
    
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('test_video', result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
