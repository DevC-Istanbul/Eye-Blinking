import cv2
import PIL
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\nuret\\Desktop\\PYTHON\\COMPUTERVISION\\Eye Blinding\\shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True :
    _,frame = cap.read()
    
    # grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #face detection
    faces=detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

    cv2.imshow("seen",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



