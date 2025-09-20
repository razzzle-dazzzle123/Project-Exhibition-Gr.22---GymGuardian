import cv2 as cv
import mediapipe as mp
import time
import math


class poseDetector():
    def __init__(self, mode= False, upperBody= False, smoothness= True, detectionConfidence= 0.5, trackingConfidence= 0.5):

        self.mode = mode
        self.upperBody = upperBody
        self.smoothness = smoothness
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smoothness, min_detection_confidence=self.detectionConfidence, min_tracking_confidence=self.trackingConfidence)


    def findPose(self, img, draw= True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    

    def findPosition(self, img, draw= True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255,0,0), cv.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw = True):
        #for landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #angle calculation
        angle  = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        #drawing
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (0,255,255), 3)
            cv.line(img, (x3, y3), (x2, y2), (0,255,255), 3)
            cv.circle(img, (x1, y1), 5, (0,0,255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0,0,255), 2)
            cv.circle(img, (x2, y2), 5, (0,0,255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0,0,255), 2)
            cv.circle(img, (x3, y3), 5, (0,0,255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0,0,255), 2)
            cv.putText(img, str(abs(int(angle))), (x2+20, y2+15), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        return angle


    

def main():
    cap = cv.VideoCapture(0)
    pTime = 0


    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        #print(lmList)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)

        cv.imshow('Img', img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()