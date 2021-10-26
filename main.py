import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pygame
import sys





def auto_piano(filename, keycolor_lower= (110, 155, 20), keycolor_upper = (130, 255, 255)):
    pygame.init()
    pygame.display.set_mode((200, 100))
    c_sound = pygame.mixer.Sound('piano C.wav')

    cap = cv.VideoCapture(str(filename))
    # cap = cv.VideoCapture(0)


    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    lower = np.array([keycolor_lower])
    upper = np.array([keycolor_upper])

    length_list = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        else:
            success, img = cap.read()
            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mask = cv.inRange(imgHSV, lower, upper)
            mask = cv.blur(mask, (7, 7))
            results = hands.process(imgRGB)
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
            try:
                if length < 40 and length_list[-1] > 40:
                    c_sound.play()
            except IndexError:
                pass

            print(length)
            length_list.append(length)
            cv.imshow('cam', img)
            cv.imshow('mask', mask)
            cv.waitKey(1)

def image_div(filename, divisions= 8):
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
    lower = np.array([170, 155, 20])
    upper = np.array([180, 255, 255])
    img = cv.imread(str(filename))
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(imgHSV, lower, upper)
    mask = cv.blur(mask, (7, 7))
    contours, heirarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 200, 20), 2)
        divs = w//divisions
        for i in range(1, divisions + 1):
            cv.circle(img, (x + divs * i - divs//2, y + 10), 5, (20, 200, 255), cv.FILLED)
            cv.line(img, (x + divs * i, y + 2), (x + divs * i, y + h - 2), (0, 0, 0), thickness= 2)
            cv.putText(img, notes[i - 1], (x + divs * i - divs//2 - 10, y + h//2 + 10), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv.imshow('grey_rec', img)
    cv.waitKey(0)


image_div('red rec.png', 8)