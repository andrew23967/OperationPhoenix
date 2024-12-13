import cv2
import numpy as np
from queue import deque
import time

def click_event(event, x, y, flags, params): 
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(x, ' ', y)
        print(thinned[y][x])

def find_nearest_white(img, target):
    nonzero = np.argwhere(img == 255)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

img = cv2.imread("images.jpg")

draw_img = img.copy()
  
gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, apertureSize=3)

kernel = np.ones((3, 3), np.uint8)

edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

a, b = cv2.connectedComponents(edges)

potential_maze_borders = []
fill_in = []

for contour in contours:
    M = cv2.moments(contour)
    
    if M["m00"] != 0:
        if cv2.contourArea(contour) > 200:
            potential_maze_borders.append((cv2.contourArea(contour), contour))
        else:
            fill_in.append(contour)

maze_border = sorted(potential_maze_borders, key=lambda x: x[0], reverse = True)[0][-1]

contour_mask = np.zeros_like(edges, dtype=np.uint8)  
cv2.fillPoly(contour_mask, [maze_border], 255)

_, black_mask = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY_INV)
cv2.drawContours(black_mask, fill_in, -1, color=1, thickness=cv2.FILLED)

intersection_mask = cv2.bitwise_and(contour_mask, black_mask)

thinned = cv2.ximgproc.thinning(intersection_mask)

#thinned = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, kernel)

start = (20, 300)
end = (200, 33)

start = find_nearest_white(thinned, start)
end = find_nearest_white(thinned, end)

thinned[thinned == 255] = [1]

for i in range(1, len(thinned) - 1):
    for ii in range(1, len(thinned[i])):
        if thinned[i][ii] == 0:
            if thinned[i-1][ii] == 1 and thinned[i][ii-1] == 1 and thinned[i-1][ii-1] == 0:
                thinned[i][ii] = 1
            elif thinned[i+1][ii] == 1 and thinned[i][ii-1] == 1 and thinned[i+1][ii-1] == 0:
                thinned[i][ii] = 1

def solve_maze(maze, start, end):
    """
    Solve the maze using BFS from start to end.

    :param maze: 2D numpy array (0 for walls, 1 for paths).
    :param start: Tuple (row, col) for the starting point.
    :param end: Tuple (row, col) for the ending point.
    :return: List of path coordinates if a path exists, None otherwise.
    """
    rows, cols = maze.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    queue = deque([(start, [start])])  # Queue of (current_position, path)
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        # If the end is reached, return the path
        if (x, y) == end:
            return path

        # Mark the current position as visited
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within bounds and valid
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [(nx, ny)]))

    # If no path is found, return None
    return None


# Solve the maze using the thinned variable
start_point = (start[0], start[1])  # Start point from your code
end_point = (end[0], end[1])        # End point from your code

path = solve_maze(thinned, start_point, end_point)
            
rgb_image = np.zeros((intersection_mask.shape[0], intersection_mask.shape[1], 3), dtype=np.uint8)
rgb_image[intersection_mask == 255] = [255, 255, 255]
rgb_image[intersection_mask == 0] = [0, 0, 0]

rgb_thinned = np.zeros((thinned.shape[0], thinned.shape[1], 3), dtype=np.uint8)
rgb_thinned[thinned == 1] = [255, 255, 255]
rgb_thinned[thinned == 0] = [0, 0, 0]

cv2.circle(draw_img, (start[1], start[0]), 5, (0, 255, 0), -1)
cv2.circle(draw_img, (end[1], end[0]), 5, (0, 0, 255), -1)

# Visualize the path
if path:
    for x, y in path:
        cv2.circle(draw_img, (y, x), 1, (255, 0, 0), -1)

cv2.imshow("maze", draw_img)
cv2.setMouseCallback('maze', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()


