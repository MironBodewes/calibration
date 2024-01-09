import matplotlib.pyplot as plt
import cv2
import os
from skimage import color, feature, io, util, measure, filters
from scipy import ndimage as ndi
#TODO shell befehl in python aufrufen und parsen?
# abbruch
# distanzmessung Wenn stift sich bewegt

FILENAME2 = "distance_test.png"
FILENAME = "calibresult.png"
IMG_TO_MEASURE_PATH = "./data/Objekt101.png"
SPACE_KEY_PRESS = 32
VIDEO_CAM=3

i = 100
cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(VIDEO_CAM)
while True:
    ret, img_orig = cam.read()
    assert ret, 'Camera read failed'
    img = img_orig.copy()
    img = cv2.putText(img, '<SPACE> to make images, <s> to stop', tuple(
        x//7 for x in img_orig.shape[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera Calibration', img)
    key = cv2.waitKey(1) & 0xFF
    if key == SPACE_KEY_PRESS:
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        i += 1
        cv2.imwrite("./data/Objekt"+str(i)+".png", img_gray)
    elif key == ord("s"):
        break

