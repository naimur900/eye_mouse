import cv2 as cv
import mediapipe as mp
# import pyautogui as pau

cap = cv.VideoCapture(0)
# face_mesh and another face_mesh must be same
face_mesh = mp.solutions.face_mesh.FaceMesh() 
    # cv.imshow('img', flipped_img_frame)
    # cv.waitKey(1)   

while True:
    flag , img_frame = cap.read()

    if flag == False:
        break

    flipped_img_frame = cv.flip(img_frame, 1)

    img_frame_height, img_frame_width, channel = flipped_img_frame.shape

    rgb_img_frame = cv.cvtColor(flipped_img_frame, cv.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_img_frame)


    for facial_landmarks in result.multi_face_landmarks:
        for i in range(468):
            pnt = facial_landmarks.landmark[i]
            x = int(pnt.x * img_frame_width)
            y = int(pnt.y * img_frame_height)
            
            cv.circle(flipped_img_frame, (x,y), 1, (0, 255, 0), -1)

    cv.imshow('img', flipped_img_frame)
    cv.waitKey(1) 