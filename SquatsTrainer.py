#Aquats Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector(detectionConfidence=0.7, trackingConfidence=0.7)

    #Angle range for a squat
    FULL_EXTENSION_ANGLE = 170
    FULL_FLEXION_ANGLE = 70

    #UI progress bar 
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650
    
    #Form correction 
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    #Squat counter variables
    count = 0
    dir = 0  #0 for up (extension), 1 for down (flexion)
    pTime = 0

    #UI smoothing
    smooth_bar = BAR_MAX_Y
    smooth_percentage = 0
    alpha = 0.2

    #Landmark visibility 
    VISIBILITY_THRESHOLD = 0.2
    leg_visible_since = None
    
    #Form checking variables
    knee_status, knee_color = "UNKNOWN", RED

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
        cv.putText(img, "For best tracking, stand at a slight angle to the camera", (150, 30), 
                   cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        primary_leg_points = [24, 26, 28]
        if landmarks_available(lmList, primary_leg_points):
            if leg_visible_since is None:
                leg_visible_since = time.time()
            
            duration_visible = time.time() - leg_visible_since

            if duration_visible > VISIBILITY_THRESHOLD:
                #Main angle calculation
                angle_raw = detector.findAngle(img, 24, 26, 28, draw=True)
                angle = _norm_angle(angle_raw)
                
                percentage = np.interp(angle, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                
                #Knee form check logic
                form_check_points = [25, 26, 27, 28] #Both knees and ankles
                if landmarks_available(lmList, form_check_points):
                    knee_dist = abs(lmList[26][1] - lmList[25][1])
                    ankle_dist = abs(lmList[28][1] - lmList[27][1])
                    
                    if knee_dist > ankle_dist * 0.8:
                        knee_status, knee_color = "GOOD", GREEN
                    else:
                        knee_status, knee_color = "KNEES INWARD", RED
                else:
                    knee_status, knee_color = "UNKNOWN", RED

                if percentage >= 95 and dir == 0 and knee_status == "GOOD":
                    count, dir = count + 0.5, 1
                if percentage <= 5 and dir == 1:
                    count, dir = count + 0.5, 0
                
                #UI Smoothing
                bar_height = np.interp(angle, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MIN_Y, BAR_MAX_Y))
                smooth_bar = int(smooth_bar * (1 - alpha) + bar_height * alpha)
                smooth_percentage = smooth_percentage * (1 - alpha) + percentage * alpha
        else:
            leg_visible_since = None
            knee_status, knee_color = "UNKNOWN", RED
        
        #UI
        #Progress Bar
        cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (100, int(smooth_bar)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage)}%', (90, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'SQUAT', (85, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        #Rep Counter and Form Indicator 
        cv.rectangle(img, (980, 530), (1230, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, f"KNEES: {knee_status}", (990, 570), cv.FONT_HERSHEY_PLAIN, 2, knee_color, 2)
        cv.putText(img, str(int(count)), (1005, 680), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        
        #FPS Counter
        cTime = time.time()
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Squat Counter', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()