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

def get_blinking_ratio(eye_points, facial_landmarks):
		# Finner fire ytterpunkter rundt øyet: venstre, høyre, top og bunn.
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
		
		# Horisontal linje og vertikal linje definerer en avstand horisontalt og vertikalt i øyet
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        ratio = hor_line_length/ver_line_length
        return ratio


        #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
		
		#cv2.polylines(frame, [left_eye_region], True, (0, 0,255), 2)
		#cv2.imshow("Eye", eye)

def get_gaze_ratio(eye_points, facial_landmarks):

	# Plasserer datapunktene rundt øyet i en "array"
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
								
    # Definerer en firkant rundt øyet, og fokuserer på denne.
    height, width, _ = frame.shape
	
	# Skiller det hvite i øyet fra irisen.
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, thershold_eye = cv2.threshold(gray_eye, 80, 255, cv2.THRESH_BINARY)
    height, width = thershold_eye.shape
	
	# Finner andel hvit på venstresida.
    left_side_threshold = thershold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
     
	# Finner andel hvit på høyresida.
    right_side_threshold = thershold_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
	
	# Bestemmer om brukeren ser til venstre, høyre eller i midten.
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white/right_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        #Detect blinking
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([37, 38, 39, 40, 41, 42], landmarks)
        right_eye_ratio = get_blinking_ratio([43, 44, 45, 46, 47, 48], landmarks)
        blinking_ratio = (left_eye_ratio + left_eye_ratio)/2
        if blinking_ratio > 5.8:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            pyautogui.press('e')


        #Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([37, 38, 39, 40, 41, 42], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([43, 44, 45, 46, 47, 48], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye)/2
        #thershold_eye = cv2.resize(thershold_eye, None, fx=5, fy=5)
        #eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        

        if gaze_ratio <= 0.89:
            cv2.putText(frame, "Right", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
            pyautogui.press('a')
        elif 0.90 < gaze_ratio < 1.5:
            cv2.putText(frame, "Center", (50, 100), font, 2, (0, 0, 255), 3)
        elif gaze_ratio >= 1.6:
            cv2.putText(frame, "Left", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)
            pyautogui.press('s')
        #cv2.imshow("Threshold", thershold_eye)
        #cv2.imshow("left", left_side_threshold)
        #cv2.imshow("right", right_side_threshold)
        #cv2.imshow("left eye", left_eye)
        #cv2.imshow("eye", eye)
        cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)

        #cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        #cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        #print(landmarks.part(36))
        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

        #showing direction
        cv2.imshow("New frame", new_frame)
        #print(face)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()