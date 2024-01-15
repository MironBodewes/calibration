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
VIDEO_CAM = 2
DIRECTORY = "./data/"
# if os.path.exists(DIRECTORY):
#     print("moving old chessboards to another folder")
#     _datetime = datetime.datetime.now()
#     shutil.move("./data/", "./databefore"+str(_datetime)+"/")
# os.mkdir(DIRECTORY)
index = 0
cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(VIDEO_CAM)


def make_images(visualize=True,index=0):
    index2=0
    while True:
        ret, img_orig = cam.read()
        assert ret, 'Camera read failed'
        img = img_orig.copy()
        img = cv2.putText(img, '<SPACE> to make images, <s> to stop', tuple(
            x//7 for x in img_orig.shape[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera Calibration', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            index+=1
            gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            if visualize == True:
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 9), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            cv2.imwrite("./data/checkerboard_"+str(index)+".png", gray)
        if key == ord("n"):
            gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            if visualize == True:
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 9), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            cv2.imwrite("./data/object_"+str(index2)+".png", gray)
            index2+=1
        elif key == ord("s"):
            break


make_images(True,index)
print(index)