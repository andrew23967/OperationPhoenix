import cv2
import numpy as np
import math

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)

def lineseg_dists(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    
    ab = b - a
    ap = p - a
    
    ab_squared = np.dot(ab, ab)
    if ab_squared == 0:
        return np.linalg.norm(p - a)
    
    t = np.dot(ap, ab) / ab_squared
    
    t = max(0, min(1, t))
    
    projection = a + t * ab
    
    distance = np.linalg.norm(p - projection)
    
    return distance

def find_ground(image):

    height, width, channels = image.shape 

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([10, 50, 100])
    upper_yellow = np.array([20, 255, 255])

    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goals = []

    for contour in contours2:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            goals.append((cv2.contourArea(contour), (cX, cY), contour))

    goals = sorted(goals, key=lambda x: x[0], reverse = True)

    contours = [g[2] for g in goals]
    goal_locs = [g[1] for g in goals]

    return goals[0][2]

def find_goals(image):

    height, width, channels = image.shape 

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([25, 120, 170])
    upper_yellow = np.array([40, 255, 255])

    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goals = []

    for contour in contours2:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            goals.append((cv2.contourArea(contour), (cX, cY), contour))

    goals = sorted(goals, key=lambda x: x[0], reverse = True)

    contours = [g[2] for g in goals]
    goal_locs = [g[1] for g in goals]

    cv2.drawContours(image, contours, -1, (255, 255, 0), 1)

    for goal in goal_locs:
        cv2.circle(image, goal, 2, (255, 0, 0), -1)

    return goal_locs

def get_orientation(image):

    height, width, channels = image.shape 

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 70, 110]) 
    upper_green = np.array([90, 170, 200])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lower_red1 = np.array([0, 100, 200])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 200])
    upper_red2 = np.array([180, 255, 255])

    maskr1 = cv2.inRange(hsv, lower_red1, upper_red1)
    maskr2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask3 = maskr1 | maskr2

    contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    
    for contour in contours:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            area = cv2.contourArea(contour)
            if area > max_area:
                side_indicator = (cX, cY)
                side_indicator_contour = contour
                max_area = area

    try:
        cv2.drawContours(image, side_indicator_contour, -1, (255, 255, 0), 3)
        cv2.circle(image, side_indicator, 5, (255, 0, 255), -1)
    except:
        pass
            
    corners = []

    for contour in contours3:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            corners.append((cv2.contourArea(contour), (cX, cY), contour))

    corners = sorted(corners, key=lambda x: x[0], reverse = True)[0:4]

    contours = [c[2] for c in corners]
    corners = [c[1] for c in corners]

    cv2.drawContours(image, contours, -1, (0, 255, 0), 1) 

    for corner in corners:
        cv2.circle(image, corner, 2, (0, 0, 0), -1)
    
    a1 = get_angle(corners[1], corners[0], corners[2])
    a2 = get_angle(corners[1], corners[0], corners[3])
    a3 = get_angle(corners[2], corners[0], corners[3])

    v0 = corners[0]

    if max([a1, a2, a3]) == a1:
        v1 = corners[1]
        v2 = corners[2]
        v3 = corners[3]

    elif max([a1, a2, a3]) == a2:
        v1 = corners[1]
        v2 = corners[3]
        v3 = corners[2]

    else:
        v1 = corners[2]
        v2 = corners[3]
        v3 = corners[1]

    d1 = lineseg_dists(side_indicator, v0, v1)
    d2 = lineseg_dists(side_indicator, v0, v2)
    d3 = lineseg_dists(side_indicator, v2, v3)
    d4 = lineseg_dists(side_indicator, v1, v3)

    if min([d1, d2, d3, d4]) == d1:
        back_seg = [v0, v1]
        opposite_seg = [v2, v3]

    elif min([d1, d2, d3, d4]) == d2:
        back_seg = [v0, v2]
        opposite_seg = [v1, v3]

    elif min([d1, d2, d3, d4]) == d3:
        back_seg = [v2, v3]
        opposite_seg = [v0, v1]

    else:
        back_seg = [v1, v3]
        opposite_seg = [v0, v2]


    middle_back = (int((back_seg[0][0] + back_seg[1][0])/2), int((back_seg[0][1] + back_seg[1][1])/2))
    middle_opposite = (int((opposite_seg[0][0] + opposite_seg[1][0])/2), int((opposite_seg[0][1] + opposite_seg[1][1])/2))
    
    forward_angle =  math.atan2(middle_opposite[1] - middle_back[1], middle_opposite[0] - middle_back[0])
        
    cv2.line(image, corners[0], v1, (255, 0, 255), 2)
    cv2.line(image, corners[0], v2, (255, 0, 255), 2)
    cv2.line(image, v1, v3, (255, 0, 255), 2)
    cv2.line(image, v2, v3, (255, 0, 255), 2)

    cv2.line(image, back_seg[0], back_seg[1], (255, 255, 0), 2)

    cx = int(sum([p[0] for p in corners])/4)
    cy = int(sum([p[1] for p in corners])/4)

    cv2.line(image, (cx, cy), (int(cx + math.cos(forward_angle) * 30), int(cy + math.sin(forward_angle) * 30)), (0, 0, 0), 2)

    forward_angle = math.degrees(forward_angle) % 360

    return (cx, cy), forward_angle
