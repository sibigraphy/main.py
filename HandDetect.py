import cv2 as cv
import numpy as np
import mediapipe as mp

cap = cv.VideoCapture('key top.mp4')

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv.imshow('cam', img)

    cv.waitKey(1)