from PIL import ImageGrab
import cv2
import numpy as np
import dlib
from math import hypot
from matplotlib import pyplot as plt
import pyautogui
import keyboard

#Screen size
sizeX, sizeY = pyautogui.size()
cap = cv2.VideoCapture(0)

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.DAT")
face_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN










while True:
    
    if keyboard.is_pressed('q'):
        screen = np.array(ImageGrab.grab(bbox=(0,0,sizeX, sizeY)))
        new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		#faces = detector(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #x, y = face.left(), face.top()
            #x1, y1 = face.right(), face.bottom()
            #cv2.rectangle(screen, (x, y), (x1, y1), (0, 255, 0), 2)
        
            cv2.circle(screen, ((x+int((w)/2)), (y+int((h)/2))), radius=3, color=(0, 0, 255), thickness=1)
        #cv2.line(screen, (x, y), (x1, y1), (0, 255, 0), thickness=1)
        #cv2.imshow("Threshold", thershold_eye)
        #cv2.imshow("left", left_side_threshold)
        #cv2.imshow("right", right_side_threshold)
        #cv2.imshow("left eye", left_eye)
        #cv2.imshow("eye", eye)
        #cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
        # Used to find middel of the webcam
            gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray_image,127,255,0)
            M = cv2.moments(thresh)
        # calculate x,y coordinate of center
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])
            #cv2.circle(screen, (cX, cY), 5, (255, 255, 255), -1)
            #DifferX = int((x+int((x1-x)/2)) - cX)
            #DifferY = int((y+int((y1-y)/2)) - cY)		
            #print((x+int((x1-x)/2)), (y+int((y1-y)/2)), DifferX, DifferY)
        #cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        #cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        #print(landmarks.part(36))
        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

        #showing direction
        #cv2.imshow("New frame", new_frame)
        #print(face)
        cv2.imshow("Frame", screen)
        
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()