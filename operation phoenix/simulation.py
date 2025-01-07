import numpy as np
import math
import time
import pygame

def signed_angle(u, v):
    """
    Compute the signed angle between two 2D vectors.
    
    Parameters:
        u: tuple or list, the first vector (u_x, u_y).
        v: tuple or list, the second vector (v_x, v_y).
    
    Returns:
        The signed angle in radians.
    """
    # Dot product and determinant
    dot = u[0] * v[0] + u[1] * v[1]
    det = u[0] * v[1] - u[1] * v[0]
    
    # Compute the signed angle using arctan2
    angle = math.atan2(det, dot)
    return angle

class Bot():
    def __init__ (self, win, start):
        self.win = win
        self.x, self.y = start
        self.speed = 0 
        self.angle = 0

    def draw(self):

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        pygame.draw.circle(self.win, (255, 0, 255), (self.x, self.y), 12)
        pygame.draw.circle(self.win, (0, 0, 0), (self.x, self.y), 12, 1)
        pygame.draw.line(self.win, (0, 0, 0), (self.x, self.y), (self.x + math.cos(self.angle) * 12, self.y + math.sin(self.angle) * 12), 1)

    def get_orientation(self):
        return np.array([self.x, self.y]), self.angle 

def run_simulation(maze_mask, start, paths, candles):
    width, height = maze_mask.shape
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("maze sim")

    surface = pygame.surfarray.make_surface(maze_mask)

    b = Bot(win, start)

    l = 0
    r = 0
    angle = 0

    path_index = 0

    current_candle = candles[path_index] 

    current_path = paths[path_index]

    spin_time = 3

    blow_out_candle = False

    min_destination_dist = 10

    run = True

    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if blow_out_candle:

            b.angle += math.radians(1)
            b.speed = 0

            if time.time() - start_time > spin_time:
                path_index += 1

                if path_index == len(paths):
                    break

                current_candle = candles[path_index] 
                current_path = paths[path_index]

                blow_out_candle = False

        else:
            position, bot_angle = b.get_orientation()

            if math.dist(current_path[0], position) < min_destination_dist:
                current_path.pop(0)

            if not len(current_path):
                blow_out_candle = True
                start_time = time.time()
                continue

            angle_to_goal = signed_angle(current_path[0] - position, (math.cos(bot_angle), math.sin(bot_angle))) #flipping order of arguments will add 180 to angle

            if angle_to_goal > math.radians(5): #counter clockwise
                b.speed = 0
                b.angle -= math.radians(1)

            elif angle_to_goal < math.radians(-5): #clockwise
                b.speed = 0
                b.angle += math.radians(1)

            else:
                b.speed = 0.15

        b.angle %= math.pi * 2

        win.fill((255,255,255))

        win.blit(surface, (0, 0))

        b.draw()

        for i, waypoint in enumerate(current_path):
            p = (i+1)/len(current_path)
            pygame.draw.circle(win, (255 * p, 0, 255 * p), waypoint, 2)

        pygame.draw.circle(win, (0,255,0), current_candle, 5)

        for candle in candles:
            pygame.draw.circle(win, (255, 0, 0), candle, 10, 1)

        if not blow_out_candle:
            pygame.draw.circle(win, (0,0,255), current_path[0], 5)
            
        pygame.display.update()