import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from pynput.keyboard import Controller, Key

keyboard = Controller()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def draw_face_rectangle(frame, shape):
    x_min = np.min(shape[:, 0])
    x_max = np.max(shape[:, 0])
    y_min = np.min(shape[:, 1])
    y_max = np.max(shape[:, 1])
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

EYE_AR_THRESHOLD = 0.25  
EYE_AR_CONSEC_FRAMES = 3 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(l_start, l_end) = (42, 48)
(r_start, r_end) = (36, 42)

cap = cv2.VideoCapture(0)
blink_counter = 0

while True:                                 
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        draw_face_rectangle(frame, shape)
                 
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESHOLD:
            blink_counter += 1
               
            print("Blink detected! Pressing space key...")
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            blink_counter = 0
     
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
