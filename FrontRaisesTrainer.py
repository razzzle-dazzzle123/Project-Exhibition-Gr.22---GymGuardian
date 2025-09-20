#Front Raises Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector(detectionConfidence=0.7, trackingConfidence=0.7)

    #Angle thresholds 
    ARM_DOWN_ANGLE = 25
    ARM_UP_ANGLE = 110

    #UI progress bar
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650
    
    #Form correction constants
    CORRECT_ELBOW_ANGLE_MIN = 100
    HORIZONTAL_DISTANCE_THRESHOLD = 300 #Max horizontal pixels between shoulder and wrist
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    #Rep counters and directions
    count_right, count_left = 0, 0
    dir_right, dir_left = 0, 0
    pTime = 0

    #UI smoothing
    smooth_bar_right, smooth_bar_left = BAR_MAX_Y, BAR_MAX_Y
    smooth_percentage_right, smooth_percentage_left = 0, 0
    alpha = 0.2
    
    #Landmark visibility persistence
    VISIBILITY_THRESHOLD = 0.2
    right_arm_visible_since = None
    left_arm_visible_since = None
    
    #Form status variables
    right_elbow_status, left_elbow_status = "UNKNOWN", "UNKNOWN"
    right_elbow_color, left_elbow_color = RED, RED
    right_raise_status, left_raise_status = "UNKNOWN", "UNKNOWN"
    right_raise_color, left_raise_color = RED, RED

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
        
        if len(lmList) == 0:
            right_arm_visible_since = None
            left_arm_visible_since = None

        #Left Arm
        right_arm_points = [24, 12, 14] #hip-shoulder-elbow
        right_form_points = [12, 14, 16] #shoulder-elbow-wrist
        if landmarks_available(lmList, right_arm_points) and landmarks_available(lmList, right_form_points):
            if right_arm_visible_since is None: right_arm_visible_since = time.time()
            duration_visible = time.time() - right_arm_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                angle_right = _norm_angle(detector.findAngle(img, *right_arm_points, draw=False))
                percentage_right = np.interp(angle_right, (ARM_DOWN_ANGLE, ARM_UP_ANGLE), (0, 100))
                
                #Elbow form check
                elbow_angle_right = _norm_angle(detector.findAngle(img, *right_form_points, draw=False))
                if elbow_angle_right > CORRECT_ELBOW_ANGLE_MIN: right_elbow_status, right_elbow_color = "GOOD", GREEN
                else: right_elbow_status, right_elbow_color = "FIX ELBOW", RED
                
                #Raise width form check
                shoulder_x, wrist_x = lmList[12][1], lmList[16][1]
                if abs(shoulder_x - wrist_x) < HORIZONTAL_DISTANCE_THRESHOLD: right_raise_status, right_raise_color = "GOOD", GREEN
                else: right_raise_status, right_raise_color = "TOO WIDE", RED
                
                #Smarter Rep Counting
                if percentage_right >= 95 and dir_right == 0 and right_elbow_status=="GOOD" and right_raise_status=="GOOD":
                    count_right, dir_right = count_right + 0.5, 1
                if percentage_right <= 5 and dir_right == 1:
                    count_right, dir_right = count_right + 0.5, 0

                #Drawing and smoothing
                bar_height_right = np.interp(angle_right, (ARM_DOWN_ANGLE, ARM_UP_ANGLE), (BAR_MAX_Y, BAR_MIN_Y))
                smooth_bar_right = int(smooth_bar_right*(1-alpha) + bar_height_right*alpha)
                smooth_percentage_right = smooth_percentage_right*(1-alpha) + percentage_right*alpha
                cv.line(img, (lmList[24][1], lmList[24][2]), (lmList[12][1], lmList[12][2]), (0,255,255), 3)
                cv.line(img, (lmList[14][1], lmList[14][2]), (lmList[12][1], lmList[12][2]), (0,255,255), 3)
                cv.putText(img, str(int(angle_right)), (lmList[12][1]+10, lmList[12][2]-10), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        else:
            right_arm_visible_since = None
            right_elbow_status, right_raise_status = "UNKNOWN", "UNKNOWN"


        #Left Arm
        left_arm_points = [23, 11, 13]
        left_form_points = [11, 13, 15]
        if landmarks_available(lmList, left_arm_points) and landmarks_available(lmList, left_form_points):
            if left_arm_visible_since is None: left_arm_visible_since = time.time()
            duration_visible = time.time() - left_arm_visible_since
            
            if duration_visible > VISIBILITY_THRESHOLD:
                angle_left = _norm_angle(detector.findAngle(img, *left_arm_points, draw=False))
                percentage_left = np.interp(angle_left, (ARM_DOWN_ANGLE, ARM_UP_ANGLE), (0, 100))

                #Elbow form check
                elbow_angle_left = _norm_angle(detector.findAngle(img, *left_form_points, draw=False))
                if elbow_angle_left > CORRECT_ELBOW_ANGLE_MIN: left_elbow_status, left_elbow_color = "GOOD", GREEN
                else: left_elbow_status, left_elbow_color = "FIX ELBOW", RED

                #Raise width form check
                shoulder_x, wrist_x = lmList[11][1], lmList[15][1]
                if abs(shoulder_x - wrist_x) < HORIZONTAL_DISTANCE_THRESHOLD: left_raise_status, left_raise_color = "GOOD", GREEN
                else: left_raise_status, left_raise_color = "TOO WIDE", RED

                #Smarter Rep Counting
                if percentage_left >= 95 and dir_left == 0 and left_elbow_status=="GOOD" and left_raise_status=="GOOD":
                    count_left, dir_left = count_left + 0.5, 1
                if percentage_left <= 5 and dir_left == 1:
                    count_left, dir_left = count_left + 0.5, 0

                #Drawing and smoothing
                bar_height_left = np.interp(angle_left, (ARM_DOWN_ANGLE, ARM_UP_ANGLE), (BAR_MAX_Y, BAR_MIN_Y))
                smooth_bar_left = int(smooth_bar_left*(1-alpha) + bar_height_left*alpha)
                smooth_percentage_left = smooth_percentage_left*(1-alpha) + percentage_left*alpha
                cv.line(img, (lmList[23][1], lmList[23][2]), (lmList[11][1], lmList[11][2]), (0,255,255), 3)
                cv.line(img, (lmList[13][1], lmList[13][2]), (lmList[11][1], lmList[11][2]), (0,255,255), 3)
                cv.putText(img, str(int(angle_left)), (lmList[11][1]-70, lmList[11][2]-10), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        else:
            left_arm_visible_since = None
            left_elbow_status, left_raise_status = "UNKNOWN", "UNKNOWN"

        #UI
        #Progress Bars
        cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3); cv.rectangle(img, (100, int(smooth_bar_right)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_right)}%', (90, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4) 
        cv.putText(img, 'RIGHT', (85, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3); cv.rectangle(img, (1100, int(smooth_bar_left)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage_left)}%', (1090, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'LEFT', (1080, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        #Rep Counters and Form Indicators
        #Right Arm UI 
        cv.rectangle(img, (50, 480), (320, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, f"ELBOW: {right_elbow_status}", (60, 520), cv.FONT_HERSHEY_PLAIN, 2, right_elbow_color, 2)
        cv.putText(img, f"WIDTH: {right_raise_status}", (60, 560), cv.FONT_HERSHEY_PLAIN, 2, right_raise_color, 2)
        cv.putText(img, str(int(count_right)), (75, 660), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        
        #Left Arm UI 
        cv.rectangle(img, (960, 480), (1230, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, f"ELBOW: {left_elbow_status}", (970, 520), cv.FONT_HERSHEY_PLAIN, 2, left_elbow_color, 2)
        cv.putText(img, f"WIDTH: {left_raise_status}", (970, 560), cv.FONT_HERSHEY_PLAIN, 2, left_raise_color, 2)
        cv.putText(img, str(int(count_left)), (985, 660), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        
        #FPS Counter
        cTime = time.time()
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Front Raises', img)

        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()