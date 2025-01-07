import cv2
import numpy as np

def rescale(image):
    offset = 320
    new_height = 700 
        
    height, width, _ = image.shape

    x1, y1, x2, y2 = offset, 0, width - offset, height

    image = image[y1:y2, x1:x2]

    height, width, _ = image.shape

    new_width = int((width/height) * new_height)

    image = cv2.resize(image, (new_width, new_height))

    return image

def crop_image(image):
    polygon_points = []
    start = None

    def draw_polygon(event, x, y, flags, param):
        nonlocal polygon_points
        nonlocal start

        if event == cv2.EVENT_LBUTTONUP:
            polygon_points.append((x, y))
            cv2.imshow('input', image)

        if event == cv2.EVENT_RBUTTONUP:
            start = (x, y)

    image = rescale(image)

    cv2.imshow('input', image)

    height, width, _ = image.shape

    #create border
    #left click for corners, right click for starting position
    cv2.setMouseCallback('input', draw_polygon)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    border_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(border_mask, [np.array(polygon_points, dtype=np.int32)], (255, 255, 255))

    return image, border_mask, start