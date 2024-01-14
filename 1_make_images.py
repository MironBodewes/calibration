'''
Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
'''
import os
import shutil
import numpy as np
import cv2
import datetime
from argparse import ArgumentParser
from glob import glob
from os import path

SPACE_KEY_PRESS = 32
DIRECTORY="./data/"
if os.path.exists(DIRECTORY):
    print("moving old chessboards to another folder")
    _datetime=datetime.datetime.now()
    shutil.move("./data/","./databefore"+str(_datetime)+"/")
# os.mkdir(DIRECTORY)
i=100
cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(2)
while True:
    ret, img_orig = cam.read()
    assert ret, 'Camera read failed'
    img = img_orig.copy()
    img = cv2.putText(img, '<SPACE> to make images, <s> to stop', tuple(x//2 for x in img_orig.shape[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('Camera Calibration', img)
    key = cv2.waitKey(1) & 0xFF
    if key == SPACE_KEY_PRESS:
        meep=cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        i+=1
        cv2.imwrite("./data/checkerboard_"+str(i)+".png",meep)
    elif key == ord("s"):
        break