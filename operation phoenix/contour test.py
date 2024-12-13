import cv2
from image_rec import get_orientation, find_goals, find_ground

camera_index = 0  # Try 0, 1, or 2
cap = cv2.VideoCapture(camera_index)

def click_event(event, x, y, flags, params): 
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(x, ' ', y)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print(hsv[y][x])

if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Accessing camera... Press 'q' to quit.")

while True:
    ret = True
    frame = cv2.imread("m9.png")#cap.read()
    frame = cv2.resize(frame, (400, 400))
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        #goal_locs = find_goals(frame)
        #center, angle = get_orientation(frame)
        ground = find_ground(frame)
    except Exception as e:
        print(e)

    cv2.imshow("iPhone Camera Feed", frame)
    cv2.setMouseCallback('iPhone Camera Feed', click_event) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
