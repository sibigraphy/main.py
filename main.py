import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pygame
import sys

pygame.init()
pygame.display.set_mode((200, 100))
c_sound = pygame.mixer.Sound('piano C.wav')

cap = cv.VideoCapture('top view detailed.mp4')

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

lower = np.array([110, 155, 20])
upper = np.array([130, 255, 255])

length_list = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    else:
        success, img = cap.read()
        imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(imgHSV, lower, upper)
        mask = cv.blur(mask, (7, 7))
        results = hands.process(imgRBG)
        indexpoints = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id_, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id_ == 8:
                        indexpoints.append((cx, cy))
                    # else:
                    #     indexpoints.append((0, 0))

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # print(indexpoints)
        contours, heirarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        midx = 0
        midy = 0
        x2, y2 = 0, 0
        x, y, w, h = 0, 0, 0, 0
        if len(contours) != 0:

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                midx, midy = (int(x + (w / 2)), int(y + (h / 2)))
                cv.circle(img, (midx, midy), 3, (0, 100, 200), cv.FILLED)
        x1, y1 = x + int(w/2), y + int(h/2)
        try:
            x2, y2 = indexpoints[0]
        except IndexError:
            pass
        cv.line(img, (x1, y1), (x2, y2), (200, 30, 100), 2)

        length = math.hypot(x2 - x1, y2 - y1)
        if length < 40:
            cv.putText(img, "Playing", (20, 30), cv.FONT_HERSHEY_PLAIN, 2, (60, 100, 190), 2)

        if length < 40 and length_list[-1] > 40:
            c_sound.play()

        print(length)
        length_list.append(length)
        cv.imshow('cam', img)
        cv.imshow('mask', mask)
        cv.waitKey(1)
