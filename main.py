import cv2
import numpy as np
import dlib
from math import hypot
import time

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_PLAIN


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio
def gaze_dedection(mask,gray,eye_region):
    cv2.polylines(mask,[eye_region],True,255,2)
    cv2.fillPoly(mask,[eye_region],255)
    eye=cv2.bitwise_and(gray,gray,mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    result_eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    return eye,threshold_eye,result_eye
is_blinking=False
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    height, width, _ = frame.shape
    right_mask = np.zeros((height, width), np.uint8)
    left_mask = np.zeros((height, width), np.uint8)
    is_face=False


    for face in faces:
        is_face=True
        print(face)
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Gaze detection
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        right_eye,threshold_eye1,eye1=gaze_dedection(right_mask,gray,right_eye_region)

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        left_eye, threshold_eye2, eye2 = gaze_dedection(left_mask, gray, left_eye_region)
        end=time.time()

    if is_face:
        cv2.imshow("Eye", eye1)
        cv2.imshow("Threshold", threshold_eye1)
        cv2.imshow("Right eye", right_eye)

        cv2.imshow("Eye2", eye2)
        cv2.imshow("Threshold2", threshold_eye2)
        cv2.imshow("Left eye", left_eye)

        if blinking_ratio > 5.7:
            is_blinking=True
            cv2.putText(frame, "BLİNKİNG", (50, 150), font, 7, (255, 0, 0))
            start = time.time()


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
if is_blinking:
    print(end - start)
cap.release()
cv2.destroyAllWindows()
