import socket
import pygame
import math
import time
import cv2
import traceback
from image_rec import get_orientation, find_goals
pygame.init()

'''
camera_index = 0  # 0, 1, or 2
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Accessing camera... Press 'q' to quit.")
'''

win = pygame.display.set_mode((400, 400))

esp32_ip = '192.168.4.1'
esp32_port = 1672

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((esp32_ip, esp32_port))


speed = 100
max_speed = 100

turn = 0

l = 0
r = 0

angle = 0

order = []

#get_locations = True

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_SPACE] and speed <= 100:
        speed += 1
    elif speed >= 4:
        speed -= 1

    if keys[pygame.K_LEFT]:
        turn -= 0.5
    elif keys[pygame.K_RIGHT]:
        turn += 0.5

    turn %= 360
        

    l = speed * math.sin(math.radians(turn))
    r = speed * math.cos(math.radians(turn))

    wr = (speed*0.5+100)

    lw = wr * math.sin(math.radians(turn - 45))
    rw = wr * -math.cos(math.radians(turn - 45))
    '''

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        if get_locations:
            goal_locs = list(find_goals(frame))

            
        center, car_angle = get_orientation(frame)

        if get_locations:
            start = center
            while len(goal_locs) > 0:
                distances = [(math.dist(start, goal), goal) for goal in goal_locs]
                closest = sorted(distances, key=lambda x: x[0])[0]
                order.append(closest[1])
                goal_locs.remove(closest[1])
                start = closest[1]
                
            get_locations = False
        
    except Exception:
        traceback.print_exc()

    for i in range(len(order)-1):
        cv2.line(frame, order[i], order[i+1], (255, 0, 255), 2)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    if not get_locations:

        if math.dist(center, order[0]) < 10:
            order.pop(0)
        
        target = order[0]
        goal_angle = math.atan2(target[1] - center[1], target[0] - center[0])

        cv2.line(frame, center, (int(center[0] + math.cos(goal_angle) * 30), int(center[1] + math.sin(goal_angle) * 30)), (0, 255, 0), 2)

        degrees_goal_angle = math.degrees(goal_angle) % 360
        
        angle = degrees_goal_angle - car_angle

        angle %= 360

        if angle < 10 or angle > 350:
            l = 50
            r = 50
        
        elif angle < 180:
            mag = 5 + angle/180 * 50
            l = mag
            r = -mag
        else:
            mag = 5 + (360 - angle)/180 * 50
            l = -mag
            r = mag

    '''

    msg = "{},{}".format(l, r)
     
    client.send(msg.encode('utf-8'))

    data = client.recv(1024)

    #cv2.imshow("iPhone Camera Feed", frame)

    msg = "{},{}".format(l, r)
         
    client.send(msg.encode('utf-8'))

    data = client.recv(1024)

    win.fill((0, 0, 0))

    pygame.draw.circle(win, (0, 255, 0), (200, 200), wr, 2)
    pygame.draw.line(win, (0, 255, 0), (200, 200), (200 + lw, 200 + rw), 4)

    pygame.display.update()

client.close()
cap.release()
cv2.destroyAllWindows()
