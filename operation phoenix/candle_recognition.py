import cv2
import math

def nothing(x):
    pass

def find_candles(candles_and_walls, min_area, max_area, min_candle_seperation):

    contours, _ = cv2.findContours(candles_and_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.namedWindow('slider')

    cv2.resizeWindow('slider', 400, 115)

    cv2.createTrackbar('min_area', 'slider', 0, 1000, nothing)
    cv2.createTrackbar('max_area', 'slider', 0, 1000, nothing)
    cv2.createTrackbar('min_candle_seperation', 'slider', 0, 1000, nothing)
    
    cv2.setTrackbarPos('min_area', 'slider', min_area)
    cv2.setTrackbarPos('max_area', 'slider', max_area)
    cv2.setTrackbarPos('min_candle_seperation', 'slider', min_candle_seperation)

    while 1:
        min_area = cv2.getTrackbarPos('min_area', 'slider')
        max_area = cv2.getTrackbarPos('max_area', 'slider')
        min_candle_seperation = cv2.getTrackbarPos('min_candle_seperation', 'slider')

        if min_area == 0:
            min_area = 1
            cv2.setTrackbarPos('min_area', 'slider', 1)

        noise_contours = []
        wall_contours = [] #currently unused
        candles = []
        candle_contours = []

        final_mask = candles_and_walls.copy()
        final_mask_show = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

        #small contours are noise, big contours are walls, everything in the middle are candles

        for contour in contours:    
            if cv2.contourArea(contour) < min_area/50:
                noise_contours.append(contour)
            elif cv2.contourArea(contour) > max_area:
                wall_contours.append(contour)
            else:
                moment = cv2.moments(contour)
                
                center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

                min_dist = float('inf')

                for candle in candles:
                    dist = math.dist(candle, center)
                    min_dist = min(dist, min_dist)

                if min_dist > min_candle_seperation:
                    candles.append(center)
                    candle_contours.append(contour)
                    cv2.drawContours(final_mask_show, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)
                else:
                    cv2.drawContours(final_mask, [contour], -1, 0, thickness=cv2.FILLED)
                    cv2.drawContours(final_mask_show, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

        cv2.drawContours(final_mask, noise_contours, -1, 0, thickness=cv2.FILLED)
        cv2.drawContours(final_mask_show, noise_contours, -1, (0, 0, 0), thickness=cv2.FILLED)

        cv2.imshow("test_image", final_mask_show)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    return final_mask, candles, candle_contours, min_area, max_area, min_candle_seperation