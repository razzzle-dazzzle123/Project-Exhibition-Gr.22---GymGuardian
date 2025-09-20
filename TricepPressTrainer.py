#Tricep Press Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector(detectionConfidence=0.7, trackingConfidence=0.7)

    #Angle thresholds
    FULL_EXTENSION_ANGLE = 170
    FULL_FLEXION_ANGLE = 45

    #UI progress bars
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650
    
    #Form correction variables
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    #Counters
    count_right, count_left = 0, 0
    dir_right, dir_left = 0, 0
    pTime = 0

    #UI smoothing
    smooth_bar_right, smooth_bar_left = BAR_MAX_Y, BAR_MAX_Y
    smooth_percentage_right, smooth_percentage_left = 0, 0
    alpha = 0.2

    #Landmark visibility 
    VISIBILITY_THRESHOLD = 0.2
    right_arm_visible_since, left_arm_visible_since = None, None
    
    posture_status_right, posture_color_right = "UNKNOWN", RED
    posture_status_left, posture_color_left = "UNKNOWN", RED

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

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        #Right arm logic
        right_arm_points = [12, 14, 16]
        if landmarks_available(lmList, right_arm_points):
            if right_arm_visible_since is None: right_arm_visible_since = time.time()
            duration_visible = time.time() - right_arm_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                shoulder_y, elbow_y = lmList[12][2], lmList[14][2]

                if elbow_y < shoulder_y:
                    posture_status_right, posture_color_right = "GOOD", GREEN
                    angle_right = _norm_angle(detector.findAngle(img, *right_arm_points, draw=True))
                    percentage_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                    
                    if percentage_right >= 95 and dir_right == 0: count_right, dir_right = count_right + 0.5, 1
                    if percentage_right <= 5 and dir_right == 1: count_right, dir_right = count_right + 0.5, 0
                    
                    bar_height_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MIN_Y, BAR_MAX_Y))
                    smooth_bar_right = int(smooth_bar_right * (1 - alpha) + bar_height_right * alpha)
                    smooth_percentage_right = smooth_percentage_right * (1 - alpha) + percentage_right * alpha
                else:
                    posture_status_right, posture_color_right = "RAISE ELBOW", RED
        else:
            right_arm_visible_since = None
            posture_status_right = "UNKNOWN"

        #Left arm logic
        left_arm_points = [11, 13, 15]
        if landmarks_available(lmList, left_arm_points):
            if left_arm_visible_since is None: left_arm_visible_since = time.time()
            duration_visible = time.time() - left_arm_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                shoulder_y, elbow_y = lmList[11][2], lmList[13][2]
                
                if elbow_y < shoulder_y:
                    posture_status_left, posture_color_left = "GOOD", GREEN
                    angle_left = _norm_angle(detector.findAngle(img, *left_arm_points, draw=True))
                    percentage_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                    
                    if percentage_left >= 95 and dir_left == 0: count_left, dir_left = count_left + 0.5, 1
                    if percentage_left <= 5 and dir_left == 1: count_left, dir_left = count_left + 0.5, 0
                    
                    bar_height_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MIN_Y, BAR_MAX_Y))
                    smooth_bar_left = int(smooth_bar_left * (1 - alpha) + bar_height_left * alpha)
                    smooth_percentage_left = smooth_percentage_left * (1 - alpha) + percentage_left * alpha
                else:
                    posture_status_left, posture_color_left = "RAISE ELBOW", RED
        else:
            left_arm_visible_since = None
            posture_status_left = "UNKNOWN"

        #UI
        #Right Arm Bar 
        cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (100, int(smooth_bar_right)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_right)}%', (90, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'RIGHT', (85, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        #Left Arm Bar 
        cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (1100, int(smooth_bar_left)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_left)}%', (1090, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'LEFT', (1080, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        #Rep Counters and Form Indicators
        #Right Arm UI 
        cv.rectangle(img, (50, 530), (320, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, f"POSTURE: {posture_status_right}", (60, 570), cv.FONT_HERSHEY_PLAIN, 2, posture_color_right, 2)
        cv.putText(img, str(int(count_right)), (75, 680), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        
        #Left Arm UI 
        cv.rectangle(img, (960, 530), (1230, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, f"POSTURE: {posture_status_left}", (970, 570), cv.FONT_HERSHEY_PLAIN, 2, posture_color_left, 2)
        cv.putText(img, str(int(count_left)), (985, 680), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)

        #FPS Counter
        cTime = time.time()
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Tricep Press', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()