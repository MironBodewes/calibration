import os
import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
# print(objp)


def my_function(draw_images: bool = True):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    directory = "./data"
    images = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(f)
            images.append(f)
    found_imgs, imgs_counter = 0, 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (7, 9), (-1, -1), criteria)
            imgpoints.append(corners2)
            found_imgs += 1
            # Draw and display the corners
            # if (draw_images == True):
            #     cv.drawChessboardCorners(img, (7, 9), corners2, ret)
            #     cv.imshow('img', img)
            #     cv.waitKey(500)

    #
    print(corners2[6][0])
    print(corners2[0][0])
    img = cv.imread(fname)
    cv.imwrite("distorted.png", img)
    x,y=corners2[0][0]
    x=int(x)
    y=int(y)
    x2,y2=corners2[6][0]
    x2=int(x2)
    y2=int(y2)
    if (draw_images == True):
        cv.line(img, (x, y), (x2, y2), 2, 3)
        cv.imshow("distorted with line", img)
        cv.waitKey(5000)
    cv.destroyAllWindows()

    print("Checker board detected {}/{}".format(found_imgs, imgs_counter))
    objpoints, imgpoints = np.array(
        objpoints, dtype=np.float32), np.array(imgpoints, dtype=np.float32)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Camera Matrix:\n', mtx)
    print('Distortion Matrix', dist)

    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    undistorted_img = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', undistorted_img)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)))

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    img = undistorted_img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (7, 9), (-1, -1), criteria)
        imgpoints.append(corners2)
        found_imgs += 1
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 9), corners2, ret)
        if (draw_images == True):
            cv.imshow('img', img)
            cv.waitKey(5000)
        cv.imwrite("undistorted.png", img)
        return corners2


corners = my_function(True)
corners = corners[:, 0, :]
print("len=", len(corners))
print("shape=", corners.shape)


def average_distance_20mm_checkerboard():
    """ calculates the average distance measured in pixels between points
    """
    columns = 9
    rows = 7
    diffs = []
    for column in range(columns):
        for row in range(rows):
            index = column*7+row
            # print(corners[index])
            if not row+1 == rows:
                abs = np.abs(corners[index]-corners[index+1])
                diff = pow(np.sum(pow(abs, 2)), 0.5)
                diffs.append(diff)
                print("diff=", diff)
            else:  # if the next item is on a different column
                pass  # skip this distance measurement
    return np.mean(diffs)


pixel_per_cm = average_distance_20mm_checkerboard()/2
print("pixel per cm=", pixel_per_cm)  # /2 because there are 20 mm per square
with open("pixel_per_cm.txt", "w+") as file:
    file.write(str(pixel_per_cm))
