import cv2
import pygame
from start_image import get_image
from crop import crop_image
from maze_recognition import interpret_maze
from simulation import run_simulation
import socket
from irl import run_real_thing
import numpy as np

pygame.init()

image = get_image()

if image is None:
    image = cv2.imread("images/img0.png")

image, border_mask, start = crop_image(image)

paths, maze_mask, inverse_maze_mask, candles, traversable_points = interpret_maze(image, border_mask, start)

image[traversable_points == 1] = (255, 0, 0)

for candle in candles:
    cv2.circle(image, candle, 10, (0, 0, 255), 1)

cv2.imshow("parsed maze", image)

run_simulation(maze_mask, start, paths, candles)

pygame.quit()
cv2.destroyWindow("parsed maze")

esp32_ip = '192.168.4.1'
esp32_port = 1672

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((esp32_ip, esp32_port))

print("connected")

run_real_thing(client, inverse_maze_mask, start, paths, candles)

client.close()

