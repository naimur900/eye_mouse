import cv2 as cv
import mediapipe as mp
import pyautogui as pau

cam = cv.VideoCapture(0)
# face_mesh and another face_mesh must be same
face_mesh = mp.solutions.face_mesh.FaceMesh() 

while True:
    _ , frame = cam.read()
    flipped_frame = cv.flip(frame, 1)
    # cv.imshow('img', flipped_frame)
    # cv.waitKey(1) 
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_faces = output.multi_face_landmarks
    # print(landmark_faces)
    if landmark_faces:
        landmark_points = landmark_faces[0].landmark
        for landmark in landmark_points:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            print(x,y)
            color = (255, 133, 233)
            cv.circle(frame, (x,y), 3, color)

    cv.imshow('img', flipped_frame)
    cv.waitKey(1) 