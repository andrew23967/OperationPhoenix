import cv2
import numpy as np

def nothing(x):
    pass

def black_in_circle(maze_mask, center, radius):
    mask = np.ones(maze_mask.shape[:2], dtype="uint8") * 255

    cv2.circle(mask, center, radius, 0, -1)

    masked_image = cv2.bitwise_or(maze_mask, mask)

    return np.any(masked_image == 0)

def remove_untraversable(traversable_points, final_mask, image, collision_radius, min_fragment_area):

    cv2.namedWindow('slider')

    cv2.resizeWindow('slider', 400, 77)

    cv2.createTrackbar('collision_radius', 'slider', 0, 1000, nothing)
    cv2.createTrackbar('min_fragment_area', 'slider', 0, 150, nothing)
    
    cv2.setTrackbarPos('collision_radius', 'slider', collision_radius)
    cv2.setTrackbarPos('min_fragment_area', 'slider', min_fragment_area)

    while 1:
        traversable_points_copy = traversable_points.copy()
        image_copy = image.copy()
        collision_radius = cv2.getTrackbarPos('collision_radius', 'slider')
        min_fragment_area = cv2.getTrackbarPos('min_fragment_area', 'slider')

        for y, row in enumerate(traversable_points_copy):
            for x, val in enumerate(row):
                if val == 255:
                    if black_in_circle(final_mask, (x, y), int(collision_radius/50)):
                        traversable_points_copy[y][x] = 0

        contours, _ = cv2.findContours(traversable_points_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #fill in the fragments cut off by the collision radius
        fragments = []
        
        for contour in contours:
            if cv2.contourArea(contour) < min_fragment_area:
                fragments.append(contour)

        cv2.drawContours(traversable_points_copy, fragments, -1, 0, thickness=cv2.FILLED)

        image_copy[traversable_points_copy == 255] = (0, 255, 0)

        cv2.imshow("test_image", image_copy)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return traversable_points_copy, collision_radius, min_fragment_area