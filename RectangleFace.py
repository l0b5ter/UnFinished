import cv2
import numpy as np
import dlib
from math import hypot
import pyautogui

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.DAT")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN










while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        
        cv2.circle(frame, ((x+int((x1-x)/2)), (y+int((y1-y)/2))), radius=3, color=(0, 0, 255), thickness=1)
        #cv2.line(frame, (x, y), (x1, y1), (0, 255, 0), thickness=1)
        #cv2.imshow("Threshold", thershold_eye)
        #cv2.imshow("left", left_side_threshold)
        #cv2.imshow("right", right_side_threshold)
        #cv2.imshow("left eye", left_eye)
        #cv2.imshow("eye", eye)
        #cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
        # Used to find middel of the webcam
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,127,255,0)
        M = cv2.moments(thresh)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        DifferX = int((x+int((x1-x)/2)) - cX)
        DifferY = int((y+int((y1-y)/2)) - cY)		
        print((x+int((x1-x)/2)), (y+int((y1-y)/2)), DifferX, DifferY)
        #cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        #cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        #print(landmarks.part(36))
        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

        #showing direction
        #cv2.imshow("New frame", new_frame)
        #print(face)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()