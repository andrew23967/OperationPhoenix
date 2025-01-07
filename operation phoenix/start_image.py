import cv2
import os
import re
import time

def get_image(port = 0):
    cap = cv2.VideoCapture(port)

    start_image = None

    screenshot_index = 0
    existing_files = os.listdir("images")
    for filename in existing_files:
        match = re.search(r"img(\d+)\.png", filename)
        if match:
            index = int(match.group(1))
            screenshot_index = max(screenshot_index, index + 1)

    last_time = 0
    delay = 1

    while True:
        ret, frame = cap.read()

        cv2.imshow("camera", frame)

        key = cv2.waitKey(3)

        if key == ord('q'):
            break

        current_time = time.time()

        if key == ord('p') and current_time - last_time > delay:
            cv2.imwrite("images\img{}.png".format(screenshot_index), frame)
            screenshot_index += 1
            start_image = frame
            cv2.imshow("most recent", frame)
            last_time = current_time

    cap.release()
    cv2.destroyAllWindows()

    return start_image