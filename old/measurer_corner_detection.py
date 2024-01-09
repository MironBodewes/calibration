import skimage
from skimage import io, color, morphology, transform, data, draw, feature
from skimage.feature import *
import matplotlib.pyplot as plt

image=io.imread("meep.png")
image=color.rgb2gray(image)
for sigma in range(2):
    for eps in [0]:
        plt.title("sigma="+str(sigma))
        something=corner_harris(image,k=0.01,eps=eps,sigma=sigma)
        plt.imshow(something)
        plt.show()
# coords = corner_peaks(corner_harris(image), min_distance=2, threshold_rel=0.01)
# coords_subpix = corner_subpix(image, coords, window_size=5)

# fig, ax = plt.subplots()
# ax.imshow(image, cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
#         linestyle='None', markersize=4)
# # plt.show()
# # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 310, 200, 0))
# plt.show()