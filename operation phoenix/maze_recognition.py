import cv2
import numpy as np
from queue import deque
import time
import itertools
import math
import itertools
from mask_generator import slider
from candle_recognition import find_candles
from collision_detection import remove_untraversable

def line_is_white(image, point1, point2, thickness):

    line_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.line(line_mask, point1, point2, 255, thickness)
    pixels = cv2.findNonZero(line_mask)

    line_pixels = [(int(pt[0][0]), int(pt[0][1])) for pt in pixels]

    for point in line_pixels:
        if image[point[1], point[0]] != 255:
            return False

    return True

def calculate_path_distance(path, distance_matrix):
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i]][path[i+1]]
    return distance

def tsp_brute_force(distance_matrix, start_city = 0):
    cities = list(range(len(distance_matrix)))
    cities.remove(start_city)
    
    min_distance = float('inf')
    best_route = None

    for path in itertools.permutations(cities):
        full_route = [start_city] + list(path)
        distance = calculate_path_distance(full_route, distance_matrix)
        
        if distance < min_distance:
            min_distance = distance
            best_route = full_route

    return best_route, min_distance

def node_distance(points):
    total_distance = 0
    for i in range(len(points) - 1):
        total_distance += math.dist(points[i], points[i+1])
    return total_distance

def find_nearest_white(img, target):
    nonzero = np.argwhere(img == 255)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return tuple(nonzero[nearest_index])

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

def solve_maze2(maze, mask, start, end, min_distance, thickness = 10):
    rows, cols = maze.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(start, [start])]) 
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if line_is_white(mask, (x, y), end, thickness) and math.dist((x, y), end) < min_distance:
            return path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [(nx, ny)]))

    return None

def interpret_maze(image, border_image, start):
    height, width, _ = image.shape

    border_mask = cv2.cvtColor(border_image, cv2.COLOR_BGR2GRAY)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    outside_blacked_hsv = cv2.bitwise_and(image_hsv, border_image)
    outside_blacked_bgr = cv2.bitwise_and(image, border_image)

    lower_hsv = np.array([0, 0, 90])
    upper_hsv = np.array([100, 60, 255])

    lower, upper = slider([outside_blacked_hsv], lower_hsv, upper_hsv)

    print(lower, upper)

    candles_and_walls_hsv = cv2.inRange(outside_blacked_hsv, lower, upper)

    lower_bgr = np.array([129, 146, 152])
    upper_bgr = np.array([255, 255, 255])

    #use both bgr and hsv mask to find candle
    lower, upper = slider([outside_blacked_bgr], lower_bgr, upper_bgr)

    print(lower, upper)

    candles_and_walls_bgr = cv2.inRange(outside_blacked_bgr, lower, upper)

    candles_and_walls = cv2.bitwise_and(candles_and_walls_hsv, candles_and_walls_bgr)
    
    min_area = 5 * 50 #scale factor for more precision with slider
    max_area = 200
    min_candle_seperation = 100

    final_mask, candles, candle_contours, min_area, max_area, min_candle_seperation = find_candles(candles_and_walls, min_area, max_area, min_candle_seperation)

    print(min_area, max_area, min_candle_seperation)

    final_mask = cv2.bitwise_not(final_mask)
    
    final_mask = cv2.bitwise_and(final_mask, border_mask)

    #thin and apply morphology to get central, traversable path
    thinned = cv2.ximgproc.thinning(final_mask)

    kernel = np.ones((5, 5), np.uint8) 

    traversable_points = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, kernel)

    #fill in diagnols
    for i in range(1, len(traversable_points) - 1):
        for ii in range(1, len(traversable_points[i])):
            if traversable_points[i][ii] == 0:
                if traversable_points[i-1][ii] == 255 and traversable_points[i][ii-1] == 255 and traversable_points[i-1][ii-1] == 0:
                    traversable_points[i][ii] = 255
                elif traversable_points[i+1][ii] == 255 and traversable_points[i][ii-1] == 255 and traversable_points[i+1][ii-1] == 0:
                    traversable_points[i][ii] = 255
    
    #remove all points on traversable_points that are within a certain radius of a black pixel
    collision_radius = 10 * 50
    min_fragment_area = 50

    traversable_points, collision_radius, min_fragment_area = remove_untraversable(traversable_points, final_mask, image, collision_radius, min_fragment_area)

    print(collision_radius, min_fragment_area)

    #get nodes and move them onto the path
    start = find_nearest_white(traversable_points, (start[1], start[0]))
    nodes = [start] + [find_nearest_white(traversable_points, (candle[1], candle[0])) for candle in candles]

    traversable_points[traversable_points == 255] = [1] #maze program has path pixels = 1 and image has them at 255 so need to convert
    
    #get paths and distances between every combination of nodes and store them in a matrix
    #a path is a set of points that tells you how to get from one node to another
    #a route is a list of nodes in the order by which they should be traversed
    path_matrix = []
    distance_matrix = []

    for node in nodes:
        sub_path_matrix = []
        sub_distance_matrix = []
        for node2 in nodes:
            path = solve_maze(traversable_points, node, node2)
            if path:
                sub_path_matrix.append(path)
                sub_distance_matrix.append(node_distance(path))
            else:
                sub_path_matrix.append([])
                sub_distance_matrix.append(float('inf'))

        path_matrix.append(sub_path_matrix)
        distance_matrix.append(sub_distance_matrix)

    #get the best route
    best_route, min_distance = tsp_brute_force(distance_matrix)

    '''
    alter path so instead of going to closest point to candle,
    just goes to the earliest point when set distance from candle
    then fix the next path so it start from the new endpoint
    '''

    paths = []
    start = nodes[0]
    ordered_candles = []

    for i in range(1, len(best_route)):
        candle = candles[best_route[i] - 1] #subtract 1 becaue candles does not have the start like nodes does
        ordered_candles.append(candle)
        candle_contour = candle_contours[best_route[i] - 1]

        mask = final_mask.copy()
        cv2.drawContours(mask, [candle_contour], -1, 255, thickness=cv2.FILLED)
        
        path = solve_maze2(traversable_points, mask, start, (candle[1], candle[0]), min_distance = 30, thickness = 10) #can alter line thickness inside of function
        paths.append(path)

        start = path[-1]
        
    #reduce the number of point in the path and reverse it
    waypoint_distance = 30
    simple_paths = []

    #flip x and y because indexing and drawing are flipped, and breakup path into waypoints
    #first level of list now represents x and second represents y
    for path in paths:
        simple_path = []
        for i, point in enumerate(path):
            if i % waypoint_distance == 0:
                simple_path.append((point[1], point[0]))

        if path[-1] not in simple_path:
            simple_path.append((path[-1][1], path[-1][0]))
            
        simple_paths.append(simple_path)


    #more flipping
    final_mask_reversed = np.rot90(final_mask, -1)

    final_mask_reversed = np.fliplr(final_mask_reversed)  

    return simple_paths, final_mask_reversed, final_mask, ordered_candles, traversable_points
