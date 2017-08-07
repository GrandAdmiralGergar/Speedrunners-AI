import opencv as cv2
import numpy as np
import tensorflow as tf
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import A, D, W, K, L, PressKey, ReleaseKey
from inception import inception_v3
import directkeys
from collections import deque
import os
import cv2
import time

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCH = 5
MODEL_NAME = 'Inceptionv3_test.model'
keytime = 0.05
timeMultiplier = 10.0

def countdown(count):
    for _ in range(count):
        print(count, '...')
        count-=1
        time.sleep(1)

def jumpleft(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Up-Left")
    PressKey(W)
    PressKey(A)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [1.,0.,1.,0.,0.]:
        ReleaseKey(A)
        ReleaseKey(W)

def jumpright(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Up-Right")
    PressKey(W)
    PressKey(D)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [0.,1.,1.,0.,0.]:
        ReleaseKey(D)
        ReleaseKey(W)

def leftslide(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Left-Slide")
    PressKey(A)
    PressKey(L)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [1.,0.,0.,0.,1.]:
        ReleaseKey(A)
        ReleaseKey(L)

def rightslide(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Right-Slide")
    PressKey(D)
    PressKey(L)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [0.,1.,0.,0.,1.]:
        ReleaseKey(D)
        ReleaseKey(L)

def leftgrapple(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Left-Grapple")
    PressKey(A)
    PressKey(K)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [1.,0.,0.,1.,0.]:
        ReleaseKey(K)
        ReleaseKey(A)

def rightgrapple(lastcommand=[0,0,0,0,0]):
    print("AI Presses: Right-Grapple")
    PressKey(D)
    PressKey(K)
    time.sleep(keytime*timeMultiplier)
    if lastcommand != [0.,1.,0.,1.,0.]:
        ReleaseKey(K)
        ReleaseKey(D)

def left(lastcommand=[0.,0.,0.,0.,0.]):
    print("AI Presses: Left")
    ReleaseKey(D)
    PressKey(A)
    #ReleaseKey(W)
    #ReleaseKey(A)
    time.sleep(keytime)
    if lastcommand != [1.,0.,0.,0.,0.]:
        ReleaseKey(A)

def right(lastcommand=[0.,0.,0.,0.,0.]):
    print("AI Presses: Right")
    ReleaseKey(A)
    PressKey(D)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(keytime)
    if lastcommand != [0.,1.,0.,0.,0.]:
        ReleaseKey(D)
    #PressKey(D)
    #PressKey(A)
    #ReleaseKey(W)

debug = True
def keys_to_output(keys):
    #[A,W,D,JMP,SLD,grapple]
    output = [0,0,0,0,0]
    if 'A' in keys:
        output[0] = 1
        if 'W' in keys:
            output[2] = 1
        if 'L' in keys:
            output[3] = 1
        if 'K' in keys:
            output[4] = 1
    elif 'D':
        output[1] = 1
        if 'W' in keys:
            output[2] = 1
        if 'L' in keys:
            output[3] = 1
        if 'K' in keys:
            output[4] = 1

    return output

def processImage(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    gray = cv2.resize(screen, (80, 60))
    return gray

def testloop():
    previouslyjumped = False
    WINDOW_WIDTH = 808
    WINDOW_HEIGHT = 481
    previousTime = time.time()
    lastcommand = [0.,0.,0.,0.,0.]

    paused = False
    while True:
        if not paused:
            screen = grab_screen([0,0,WINDOW_WIDTH,WINDOW_HEIGHT])
            screen = processImage(screen)


            previousTime = time.time()

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0]
            moves = list(np.around(prediction))
            print(prediction)
            jump = False
            slide = False
            grapple = False
            firstjumpthreshold = 0.2
            secondjumpthreshold = 0.2
            if prediction[2] >= (secondjumpthreshold if previouslyjumped else firstjumpthreshold):
                jump = True
            if prediction[3] >= 0.1:
                slide = True

            if prediction[4] >= 0.2:
                grapple = True

            if moves[0] == [1.]:
                if jump:

                    jumpleft(lastcommand)
                    previouslyjumped = True
                elif slide:
                    leftslide(lastcommand)
                elif grapple:
                    leftgrapple(lastcommand)
                else:

                    left(lastcommand)
            if moves[1] == [1.]:
                if jump:

                    jumpright(lastcommand)
                    previouslyjumped = True
                elif slide:
                    rightslide(lastcommand)
                elif grapple:
                    rightgrapple(lastcommand)
                else:
                    right(lastcommand)

                lastcommand = moves
                previouslyjumped = jump
                #jump = False

        keys = key_check()
        if 'T' in keys:
            paused = True
            os.system('cls')
            time.sleep(1)

        if 'Y' in keys:
            os.system('cls')
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            paused = False

model = inception_v3(WIDTH, HEIGHT, 3, LR, output=5)
model.load(MODEL_NAME)

print('Begin in a few seconds..')
countdown(3)
testloop()