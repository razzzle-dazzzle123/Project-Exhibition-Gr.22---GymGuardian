#Bicep Curls Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector()

    #Angle range for a bicep curl
    FULL_EXTENSION_ANGLE = 160
    FULL_FLEXION_ANGLE = 45
    
    #Vertical progress bars
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650

    #Variables for BOTH Arms
    #Left Arm
    count_left = 0
    dir_left = 0  # 0 for extension, 1 for flexion
    
    #Right Arm
    count_right = 0
    dir_right = 0  # 0 for extension, 1 for flexion
    
    pTime = 0

    #Smoothing variables for Right Arm UI
    smooth_bar_right = BAR_MAX_Y
    smooth_percentage_right = 0

    #Smoothing variables for Left Arm UI
    smooth_bar_left = BAR_MAX_Y
    smooth_percentage_left = 0

    alpha = 0.2  #smoothing factor (lower = smoother)

    #Normalize angles
    def _norm_angle(a):
        a = a % 360.0
        if a > 180.0:
            a = 360.0 - a
        return a

    #Safely check if required landmarks exist
    def landmarks_available(lmList, points):
        return all(p < len(lmList) for p in points)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img = cv.resize(img, (1280, 720))
        #Flip
        img = cv.flip(img, 1)

        #Find the pose and landmarks
        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            #Right Arm Logic 
            if landmarks_available(lmList, [12, 14, 16]):
                angle_right_raw = detector.findAngle(img, 12, 14, 16)
                angle_right = _norm_angle(angle_right_raw)
                percentage_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                bar_height_right = np.interp(angle_right, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MIN_Y, BAR_MAX_Y))
                
                #Smooth right side UI values
                smooth_bar_right = int(smooth_bar_right * (1 - alpha) + bar_height_right * alpha)
                smooth_percentage_right = smooth_percentage_right * (1 - alpha) + percentage_right * alpha
                
                #Check for right arm rep completion
                if percentage_right >= 95 and dir_right == 0:
                    count_right += 0.5
                    dir_right = 1
                if percentage_right <= 5 and dir_right == 1:
                    count_right += 0.5
                    dir_right = 0

            #Left Arm Logic
            if landmarks_available(lmList, [11, 13, 15]):
                angle_left_raw = detector.findAngle(img, 11, 13, 15)
                angle_left = _norm_angle(angle_left_raw)
                percentage_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (100, 0))
                bar_height_left = np.interp(angle_left, (FULL_FLEXION_ANGLE, FULL_EXTENSION_ANGLE), (BAR_MIN_Y, BAR_MAX_Y))
                
                #Smooth left side UI values
                smooth_bar_left = int(smooth_bar_left * (1 - alpha) + bar_height_left * alpha)
                smooth_percentage_left = smooth_percentage_left * (1 - alpha) + percentage_left * alpha
                
                #Check for left arm rep completion
                if percentage_left >= 95 and dir_left == 0:
                    count_left += 0.5
                    dir_left = 1
                if percentage_left <= 5 and dir_left == 1:
                    count_left += 0.5
                    dir_left = 0

            # Right Side UI (for Right Arm)
            # cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3)
            # cv.rectangle(img, (1100, int(smooth_bar_right)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
            # cv.putText(img, f'{int(smooth_percentage_right)}%', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            # cv.putText(img, 'RIGHT', (1080, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3)
            cv.rectangle(img, (100, int(smooth_bar_right)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
            cv.putText(img, f'{int(smooth_percentage_right)}%', (100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            cv.putText(img, 'LEFT', (90, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
            # Left Side UI (for Left Arm)
            # cv.rectangle(img, (100, BAR_MIN_Y), (175, BAR_MAX_Y), (0, 255, 0), 3)
            # cv.rectangle(img, (100, int(smooth_bar_left)), (175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
            # cv.putText(img, f'{int(smooth_percentage_left)}%', (100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            # cv.putText(img, 'LEFT', (90, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3)
            cv.rectangle(img, (1100, int(smooth_bar_left)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
            cv.putText(img, f'{int(smooth_percentage_left)}%', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
            cv.putText(img, 'RIGHT', (1080, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            #Rep Counters 
            cv.putText(img, str(int(count_left)), (1050, 670), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 15)
            cv.putText(img, str(int(count_right)), (45, 670), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 15)

        else:
            #Show warning if no landmarks detected
            cv.putText(img, "Move into frame!", (500, 350), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

        #FPS Counter
        cTime = time.time()
        #Handle potential division by zero
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Bicep Curls', img) 

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()