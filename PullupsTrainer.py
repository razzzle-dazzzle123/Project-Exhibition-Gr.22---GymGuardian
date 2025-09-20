#Pullups Module
import cv2 as cv
import numpy as np
import time
import PoseEstimationModule as pm

def run():
    cap = cv.VideoCapture(0)
    detector = pm.poseDetector(detectionConfidence=0.7, trackingConfidence=0.7)

    PULLUP_THRESHOLD = 75 #Required pixel displacement to count a rep

    #UI progress bar 
    BAR_MIN_Y = 100
    BAR_MAX_Y = 650
    
    YELLOW = (0, 255, 255)

    #Rep counter 
    count = 0
    dir = 0 
    start_y = None 
    pTime = 0

    #UI smoothing
    smooth_bar = BAR_MAX_Y
    smooth_percentage = 0
    alpha = 0.2

    def landmarks_available(lmList, points):
        return all(p < len(lmList) for p in points)

    def midpoint(lmList, p1, p2):
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv.resize(img, (1280, 720))
        img = cv.flip(img, 1)

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        
        displacement = 0
        core_body_points = [11, 12] 
        if landmarks_available(lmList, core_body_points):
            sh_mid_x, sh_mid_y = midpoint(lmList, 11, 12)

            if start_y is None:
                start_y = sh_mid_y
            
            displacement = max(0, start_y - sh_mid_y)
            
            if dir == 0 and displacement >= PULLUP_THRESHOLD:
                count += 1
                dir = 1 

            elif dir == 1 and displacement < 20:
                dir = 0
            
            #UI
            percentage = np.interp(displacement, (0, PULLUP_THRESHOLD + 20), (0, 100)) 
            bar_height = np.interp(percentage, (0, 100), (BAR_MAX_Y, BAR_MIN_Y))
            smooth_bar = int(smooth_bar * (1 - alpha) + bar_height * alpha)
            smooth_percentage = smooth_percentage * (1 - alpha) + percentage * alpha

            cv.putText(img, f"Disp: {int(displacement)}px", (sh_mid_x - 70, sh_mid_y - 20), cv.FONT_HERSHEY_PLAIN, 2, YELLOW, 2)
            l_sh_x, l_sh_y = lmList[11][1:]; r_sh_x, r_sh_y = lmList[12][1:]
            cv.circle(img, (l_sh_x, l_sh_y), 8, YELLOW, cv.FILLED); cv.circle(img, (r_sh_x, r_sh_y), 8, YELLOW, cv.FILLED)
            cv.line(img, (l_sh_x, l_sh_y), (r_sh_x, r_sh_y), YELLOW, 3)

        else:
            start_y = None #Reset if user leaves frame

        #UI
        cv.rectangle(img, (1100, BAR_MIN_Y), (1175, BAR_MAX_Y), (0, 255, 0), 3)
        cv.rectangle(img, (1100, int(smooth_bar)), (1175, BAR_MAX_Y), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(smooth_percentage)}%', (1090, 90), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv.putText(img, 'PULL-UPS', (1050, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv.rectangle(img, (50, 580), (320, 700), (0, 0, 0), cv.FILLED)
        cv.putText(img, str(int(count)), (75, 680), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
        
        cTime = time.time()
        if (cTime - pTime) > 0:
            fps = 1 / (cTime - pTime)
        else:
            fps = 0
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (520, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow('GymGuardian - Pullups', img)

        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()