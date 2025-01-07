import cv2
import numpy as np
import math
import time
from crop import rescale
from bot_recognition import get_orientation

def signed_angle(u, v):
    dot = u[0] * v[0] + u[1] * v[1]
    det = u[0] * v[1] - u[1] * v[0]
    
    angle = math.atan2(det, dot)
    return angle 

def run_real_thing(client, inverse_maze_mask, paths, candles, port = 0):

    cap = cv2.VideoCapture(port)

    l = 0
    r = 0
    position = np.array([0, 0])
    bot_angle = 0
    path_index = 0
    current_candle = candles[path_index] 
    current_path = paths[path_index]
    spin_time = 3
    blow_out_candle = False
    min_destination_dist = 10

    while True:

        if blow_out_candle:

            l = 10
            r = -10

            if time.time() - start_time > spin_time:
                path_index += 1

                if path_index == len(paths): #done
                    break

                current_candle = candles[path_index] 
                current_path = paths[path_index]

                l = 0
                r = 0

                blow_out_candle = False

        else:
            ret, frame = cap.read()

            frame = rescale(frame)

            frame = cv2.bitwise_and(frame, frame, mask = inverse_maze_mask)

            try:
                position, bot_angle = get_orientation(frame)
            except:
                pass

            if math.dist(current_path[0], position) < min_destination_dist:
                current_path.pop(0)

            if not len(current_path):
                blow_out_candle = True
                start_time = time.time()
                continue

            angle_to_goal = signed_angle(current_path[0] - position, (math.cos(bot_angle), math.sin(bot_angle))) #flipping order of arguments will add 180 to angle

            if angle_to_goal > math.radians(5): #counter clockwise
                l = 10
                r = -10

            elif angle_to_goal < math.radians(-5): #clockwise
                l = -10
                r = 10

            else:
                l = 40
                r = 40

        msg = "{},{}".format(l, r)

        client.send(msg.encode('utf-8'))

        data = client.recv(1024)
    
    cap.release()
    cv2.destroyAllWindows()