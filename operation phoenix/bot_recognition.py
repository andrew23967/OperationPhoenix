import cv2
import numpy as np

def biggest_contour(image_hsv, lower, upper):
    mask = cv2.inRange(image_hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)

    moment = cv2.moments(biggest)
    center = np.array([int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])])

    return center

def get_orientation(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 0, 135])
    upper_green = np.array([35, 255, 255])

    green = biggest_contour(image_hsv, lower_green, upper_green)

    lower_red = np.array([167, 136, 182])
    upper_red = np.array([194, 255, 255])

    red = biggest_contour(image_hsv, lower_red, upper_red)

    center = (green + red) / 2
    angle = np.arctan2(green[1] - red[1], green[0] - red[0])

    return center, angle


