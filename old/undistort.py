'''
Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
'''
import numpy as np
import cv2
from argparse import ArgumentParser
from glob import glob
from os import path

SPACE_KEY_PRESS = 32

def read_img_from_camera(cam):
    while True:
        ret, img_orig = cam.read()
        assert ret, 'Camera read failed'
        img = img_orig.copy()
        img = cv2.putText(img, '<SPACE>', tuple(x//2 for x in img_orig.shape[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Camera Calibration', img)
        key = cv2.waitKey(1) & 0xFF
        if key == SPACE_KEY_PRESS:
            break
    return cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

a_parser = ArgumentParser("Camera Calibration")
a_parser.add_argument('--checker_board_size', default='7,9', help='Inner squares in checker board in "row,col" format')
a_parser.add_argument('--imgs_path', default=None, help="Path where jpgs/pngs are stored. Skip to live run using camera feed")
a_parser.add_argument('--debug', default=False, action='store_true', help="Write point projected images")
a_parser.add_argument('--goal', default=30, help="Max no of imgs for images to consider for calibration", type=int)
args = a_parser.parse_args()

# cornerSubPix termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

checker_board_size = tuple( int(x) for x in args.checker_board_size.split(',') )
assert len(checker_board_size) == 2, 'Invalid checker board size'

object_points_coord = np.zeros((checker_board_size[0] * checker_board_size[1], 3))
object_points_coord[:,:2] = np.mgrid[:checker_board_size[0], :checker_board_size[1]].T.reshape(-1, 2)

if args.imgs_path:
    imgs = glob(path.join(args.imgs_path, '*.jpg'))
    imgs.extend(glob(path.join(args.imgs_path, '*.png')))
    assert len(imgs) > 0, 'Images not found in the imgs_path'
else:
    cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
found_imgs, imgs_counter = 0, 0
objpoints, imgpoints = [], []
while True:
    if args.imgs_path:
        img = cv2.imread(imgs[imgs_counter], cv2.IMREAD_GRAYSCALE)
    else:
        img = read_img_from_camera(cam)
    imgs_counter += 1
    # Find checker board corners
    ret, corners = cv2.findChessboardCorners(img, checker_board_size, None)
    if not ret: continue
    # Refine checker board corners
    corners = cv2.cornerSubPix(img, corners, (7,7), (-1,-1), criteria)
    # Add the found checker board points to list
    objpoints.append(object_points_coord)
    imgpoints.append(corners.reshape(-1, 2))
    found_imgs += 1

    cv2.drawChessboardCorners(img, checker_board_size, corners, ret)
    cv2.imshow('Camera Calibration', img)
    cv2.waitKey(500)
    if args.debug: cv2.imwrite('./__out_{}.jpg'.format(found_imgs), img)

    if found_imgs >= args.goal or (args.imgs_path and imgs_counter >= len(imgs)): break

cv2.destroyAllWindows()

print("Checker board detected {}/{}".format(found_imgs, imgs_counter))
objpoints, imgpoints = np.array(objpoints, dtype=np.float32), np.array(imgpoints, dtype=np.float32)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
print('Camera Matrix', mtx)
print('Distortion Mat', dist)

img = cv2.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )