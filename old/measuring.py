from skimage import measure, color, filters
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from skimage import io, color


filename = "distance_test.png"


def measure1():
    print(os.getcwd())
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('LIVE FRAME!', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Save it to some location
    cv2.imwrite(filename, frame)


def measure2():
    # Read Image
    image = cv2.imread(filename)
    image = image[:300]
    object_width = int(input("Enter the width of your object: "))
    object_height = int(input("Enter the height of your object: "))

    # Find Corners

    def find_centroids(dst):
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100,
                    0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids[1:]), (5, 5),
                                   (-1, -1), criteria)
        return corners

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # Get coordinates of the corners.
    corners = find_centroids(dst)

    for i in range(0, len(corners)):
        print("Pixels found for this object are:", corners[i])
        image[dst > 0.1*dst.max()] = [0, 0, 255]
        cv2.circle(image, (int(corners[i, 0]), int(
            corners[i, 1])), 7, (0, 255, 0), 2)

    for corner in corners:
        image[int(corner[1]), int(corner[0])] = [0, 0, 255]

    a = len(corners)
    print("Number of corners found:", a)

    # List to store pixel difference.
    distance_pixel = []

    # List to store mm distance.
    distance_mm = []
    P1 = corners[0]
    P2 = corners[1]
    P3 = corners[2]
    P4 = corners[3]

    P1P2 = cv2.norm(P2-P1)
    P1P3 = cv2.norm(P3-P1)
    P2P4 = cv2.norm(P4-P2)
    P3P4 = cv2.norm(P4-P3)

    pixelsPerMetric_width1 = P1P2 / object_width
    pixelsPerMetric_width2 = P3P4 / object_width
    pixelsPerMetric_height1 = P1P3 / object_height
    pixelsPerMetric_height2 = P2P4 / object_height

    # Average of PixelsPerMetric
    pixelsPerMetric_avg = pixelsPerMetric_width1 + pixelsPerMetric_width2 + \
        pixelsPerMetric_height1 + pixelsPerMetric_height2

    pixelsPerMetric = pixelsPerMetric_avg / 4
    print(pixelsPerMetric)
    P1P2_mm = P1P2 / pixelsPerMetric
    P1P3_mm = P1P3 / pixelsPerMetric
    P2P4_mm = P2P4 / pixelsPerMetric
    P3P4_mm = P3P4 / pixelsPerMetric

    distance_mm.append(P1P2_mm)
    distance_mm.append(P1P3_mm)
    distance_mm.append(P2P4_mm)
    distance_mm.append(P3P4_mm)

    distance_pixel.append(P1P2)
    distance_pixel.append(P1P3)
    distance_pixel.append(P2P4)
    distance_pixel.append(P3P4)

    print(distance_pixel)
    print(distance_mm)
    print(len(corners))
    img = io.imread("meep.png")
    plt.plot(corners[:, 0], corners[:, 1], "rx")
    plt.imshow(img)
    plt.show()

from skimage import morphology
def measure3():
    img = io.imread(filename)
    img_gray=color.rgb2gray(img)
    # array=filters.apply_hysteresis_threshold(img,60,best_threshold)
    thresh = filters.threshold_otsu(img_gray)
    binary = img_gray > thresh
    plt.imshow(binary, cmap="gray")
    plt.colorbar()
    label, num = measure.label(binary, return_num=True)
    blobs = color.label2rgb(label, img_gray)
    print(num)
    plt.show()
    plt.imshow(blobs)
    plt.show()
    # YOUR CODE HERE
    print("orientation 0 is 90°.\n -0.2 is about 75° and 0.2 is about 105°")


# measure1()
# measure2()
measure3()

