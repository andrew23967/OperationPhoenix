import cv2
import numpy as np
from queue import deque
from image_rec import find_ground
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

def find_nearest_black(img, target):
    nonzero = np.argwhere(img == 0)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    return min(distances)

def solve_maze(maze, start, end):
    rows, cols = maze.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(start, [start])]) 
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == end:
            return path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [(nx, ny)]))

    return None

img = cv2.imread("m60.png")

h, w, channels = img.shape

width = 200
height = int((h/w) * width)

img = cv2.resize(img, (width, height))

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

contour = find_ground(img)

lower_range = np.array([10, 50, 100])
upper_range = np.array([20, 255, 255])

mask = np.zeros_like(img_hsv[:, :, 0], dtype=np.uint8)

cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

result_mask = cv2.inRange(img_hsv, lower_range, upper_range) 
final_mask = cv2.bitwise_and(result_mask, result_mask, mask=mask)  

binary_output = np.zeros_like(final_mask, dtype=np.uint8)
binary_output[final_mask > 0] = 255

contours, _ = cv2.findContours(cv2.bitwise_not(binary_output), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

border_size = 15

for i in range(0, height - 1):
    for ii in range(0, width - 1):
        if i <= border_size or i >= height - border_size or ii <= border_size or ii >= width - border_size:
            binary_output[i][ii] = 0

candles = []
fill_in = []

min_area = 6.75/2

for contour in contours:
    if cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < 40:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            candles.append((cX, cY))
        
    if cv2.contourArea(contour) < min_area:
        fill_in.append(contour)

for candle in candles:
    cv2.circle(img, candle, 10, (255, 0, 255), 1)

cv2.drawContours(binary_output, fill_in, -1, 255, thickness=cv2.FILLED)


for i in range(0, height):
    for ii in range(0, width):
        if binary_output[i][ii] == 255:
            pass
            
thinned = cv2.ximgproc.thinning(binary_output)

kernel = np.array([3, 3])

thinned = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, kernel)

start = (42, 22)
end = (candles[2][1], candles[2][0])

start = find_nearest_white(thinned, start)
end = find_nearest_white(thinned, end)

start_point = (start[0], start[1])  
end_point = (end[0], end[1]) 

thinned[thinned == 255] = [1]

for i in range(1, len(thinned) - 1):
    for ii in range(1, len(thinned[i])):
        if thinned[i][ii] == 0:
            if thinned[i-1][ii] == 1 and thinned[i][ii-1] == 1 and thinned[i-1][ii-1] == 0:
                thinned[i][ii] = 1
            elif thinned[i+1][ii] == 1 and thinned[i][ii-1] == 1 and thinned[i+1][ii-1] == 0:
                thinned[i][ii] = 1      

path = solve_maze(thinned, start_point, end_point)

rgb_thinned = np.zeros((thinned.shape[0], thinned.shape[1], 3), dtype=np.uint8)
rgb_thinned[thinned == 1] = [255, 255, 255]
rgb_thinned[thinned == 0] = [0, 0, 0]

cv2.circle(img, (start[1], start[0]), 5, (0, 255, 0), -1)
cv2.circle(img, (end[1], end[0]), 5, (0, 0, 255), -1)

if path:
    for x, y in path:
        cv2.circle(img, (y, x), 1, (255, 0, 0), -1)

cv2.imshow("bit", binary_output)
cv2.imshow("thinned", rgb_thinned)
cv2.imshow("maze", img)
cv2.setMouseCallback('maze', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

