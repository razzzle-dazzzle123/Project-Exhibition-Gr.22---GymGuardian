#Lunges Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector(detectionConfidence=0.7, trackingConfidence=0.7)

    #Angle thresholds for lunges
    FULL_EXTENSION_ANGLE = 170
    FULL_FLEXION_ANGLE = 70

    #UI progress bar 
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650
    
    #Form correction variables
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    #Rep counters
    count = 0
    dir_right, dir_left = 0, 0
    pTime = 0

    #UI smoothing
    smooth_bar_right, smooth_bar_left = BAR_MAX_Y, BAR_MAX_Y
    smooth_percentage_right, smooth_percentage_left = 0, 0
    alpha = 0.2

    #Landmark visibility
    VISIBILITY_THRESHOLD = 0.2
    right_leg_visible_since, left_leg_visible_since = None, None
    
    #Form checking variables
    knee_status_right, knee_color_right = "GOOD", GREEN
    knee_status_left, knee_color_left = "GOOD", GREEN

    def _norm_angle(a):
        a = a % 360.0
        if a > 180.0:
            a = 360.0 - a
        return a

    def landmarks_available(lmList, points):
        return all(p < len(lmList) for p in points)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv.resize(img, (1280, 720))
        img = cv.flip(img, 1)

        cv.putText(img, "Stand at a slight angle to the camera for best tracking", (150, 30), 
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        #Right leg logic
        right_leg_points = [24, 26, 28]
        if landmarks_available(lmList, right_leg_points):
            if right_leg_visible_since is None: right_leg_visible_since = time.time()
            duration_visible = time.time() - right_leg_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                angle_right = _norm_angle(detector.findAngle(img, *right_leg_points, draw=True))
                percentage_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                
                if percentage_right > 80:
                    knee_x, ankle_x = lmList[26][1], lmList[28][1]
                    if knee_x > ankle_x: knee_status_right, knee_color_right = "GOOD", GREEN
                    else: knee_status_right, knee_color_right = "KNEE FORWARD", RED
                elif percentage_right < 20: knee_status_right, knee_color_right = "GOOD", GREEN
                
                if percentage_right >= 95 and dir_right == 0 and knee_status_right == "GOOD": count, dir_right = count + 0.5, 1
                if percentage_right <= 5 and dir_right == 1: count, dir_right = count + 0.5, 0
                
                bar_height_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MAX_Y, BAR_MIN_Y))
                smooth_bar_right = int(smooth_bar_right * (1 - alpha) + bar_height_right * alpha)
                smooth_percentage_right = smooth_percentage_right * (1 - alpha) + percentage_right * alpha
        else:
            right_leg_visible_since = None

        #Left leg logic
        left_leg_points = [23, 25, 27]
        if landmarks_available(lmList, left_leg_points):
            if left_leg_visible_since is None: left_leg_visible_since = time.time()
            duration_visible = time.time() - left_leg_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                angle_left = _norm_angle(detector.findAngle(img, *left_leg_points, draw=True))
                percentage_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                
                if percentage_left > 80:
                    knee_x, ankle_x = lmList[25][1], lmList[27][1]
                    if knee_x > ankle_x: knee_status_left, knee_color_left = "GOOD", GREEN
                    else: knee_status_left, knee_color_left = "KNEE FORWARD", RED
                elif percentage_left < 20: knee_status_left, knee_color_left = "GOOD", GREEN

                if percentage_left >= 95 and dir_left == 0 and knee_status_left == "GOOD": count, dir_left = count + 0.5, 1
                if percentage_left <= 5 and dir_left == 1: count, dir_left = count + 0.5, 0

                bar_height_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MAX_Y, BAR_MIN_Y))
                smooth_bar_left = int(smooth_bar_left * (1 - alpha) + bar_height_left * alpha)
                smooth_percentage_left = smooth_percentage_left * (1 - alpha) + percentage_left * alpha
        else:
            left_leg_visible_since = None

        #UI
        #Right Leg Bar 
        cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (100, int(smooth_bar_right)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_right)}%', (90, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'RIGHT', (85, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.putText(img, knee_status_right, (60, BAR_MAX_Y + 40), cv.FONT_HERSHEY_PLAIN, 2, knee_color_right, 2)

        #Left Leg Bar 
        cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (1100, int(smooth_bar_left)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_left)}%', (1090, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'LEFT', (1080, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.putText(img, knee_status_left, (1050, BAR_MAX_Y + 40), cv.FONT_HERSHEY_PLAIN, 2, knee_color_left, 2)
        
        #Total Rep Counter
        cv.rectangle(img, (510, 580), (770, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, str(int(count)), (535, 680), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        cv.putText(img, "REPS", (570, 560), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        #FPS Counter
        cTime = time.time()
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Lunges', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()