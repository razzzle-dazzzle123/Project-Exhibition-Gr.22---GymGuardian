#Pushups Module 
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def run():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    neck_status = "Unknown"
    lower_back_status = "Unknown"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1200, 820))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                #Right side
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                #Angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                neck_angle = calculate_angle(ear, shoulder, hip)
                lower_back_angle = calculate_angle(shoulder, hip, ankle)

                #Progress bar mapping 
                percentage = np.interp(elbow_angle, (90, 170), (100, 0))
                bar = np.interp(elbow_angle, (90, 170), (650, 100))

                #Pushup counter logic
                if elbow_angle > 160:
                    stage = "up"
                if elbow_angle < 100 and stage == "up":
                    stage = "down"
                    counter += 1

                #Neck correctness
                if 150 <= neck_angle <= 170:
                    neck_status = "Good"
                    neck_color = (0, 255, 0)
                else:
                    neck_status = "Fix Neck"
                    neck_color = (0, 0, 255)

                #Lower back correctness
                if 160 <= lower_back_angle <= 180:
                    lower_back_status = "Good"
                    lower_back_color = (0, 255, 0)
                else:
                    lower_back_status = "Fix Lower Back"
                    lower_back_color = (0, 0, 255)

                #Progress bar
                cv2.rectangle(image, (1100, 100), (1175, 650), (0, 255, 0), 3)
                cv2.rectangle(image, (1100, int(bar)), (1175, 650), (0, 255, 0), -1)
                cv2.putText(image, f'{int(percentage)} %', (1070, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'NECK: {neck_status}', (800, 650),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, neck_color, 3, cv2.LINE_AA)
                cv2.putText(image, f'LOWER BACK: {lower_back_status}', (800, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, lower_back_color, 3, cv2.LINE_AA)

            except:
                pass

            # Display reps 
            cv2.rectangle(image, (0, 0), (320, 100), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'STAGE', (160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, stage if stage else "-",
                        (160, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Pushup Counter (Side View with Neck & Lower Back Check)", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()