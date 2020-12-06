import cv2
import numpy as np
import dlib
from math import hypot
import pyautogui

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.DAT")
screenx, screeny = pyautogui.size()

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        ratio = hor_line_length/ver_line_length
        return ratio

def get_gaze(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    #cv2.polylines(frame, [left_eye_region], True, (0, 0,255), 2)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    eye = frame[min_y: max_y, min_x: max_x]
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, thershold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    return thershold_eye, eye

def get_gaze_contours(left_eye_threshold, left_eye, height):
    rows, cols, _ = left_eye.shape
    #eye = cv2.resize(eye, None, fx=5, fy=5)
    #thershold_eye = cv2.resize(thershold_eye, None, fx=5, fy=5)
    contours, _ = cv2.findContours(left_eye_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(left_eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(left_eye, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(left_eye, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)
        cv2.circle(left_eye, (x + int(w/2), y + int(h/2)), 3, (0, 0, 255), 1)
        #cv2.drawContours(eye, [cnt], -1, (0, 0, 255, 3))
        #print(x + int(w/2), y + int(h/2))
        #cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
        cv2.putText(new_frame, str(x + int(w/2)), (50, height), font, 2, (0, 0, 255), 3)
        cv2.putText(new_frame, str(y + int(h/2)), (150, height), font, 2, (0, 0, 255), 3)
        break
    return

while True:
    _, frame = cap.read()
    new_frame = np.zeros((300, 300, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)


        #Detect blinking
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + left_eye_ratio)/2
        if blinking_ratio > 5.8:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))


        #Gaze detection
        left_eye_threshold, left_eye = get_gaze([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_threshold, right_eye = get_gaze([42, 43, 44, 45, 46, 47], landmarks)
        
        #Contours for each eye
        get_gaze_contours(left_eye_threshold, left_eye, 150)
        get_gaze_contours(right_eye_threshold, right_eye, 250)

        #Zoom in on eye, bigger frame^^
        right_eye_threshold = cv2.resize(right_eye_threshold, None, fx=8, fy=8)
        left_eye = cv2.resize(left_eye, None, fx=8, fy=8)
        right_eye = cv2.resize(right_eye, None, fx=8, fy=8)
        #print(screenx, screeny)
        #new_frame = cv2.resize(new_frame, None, fx=screenx, fy=screeny)
        #cv2.imshow("Eye", eye)
        cv2.imshow("New frame", new_frame)
        cv2.imshow("Threshold", right_eye_threshold)
        cv2.imshow("Left eye", right_eye)
        #cv2.imshow("Right eye", right_eye)
        #print(landmarks.part(36))
        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)
        


        #print(face)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
cap.reslease()
cv2.destroyAllWindows()