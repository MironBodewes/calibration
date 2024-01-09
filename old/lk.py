import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os
import cv2
from cv2.typing import Size,MatLike
directory="/home/mirondebian/Pictures/Webcam"

images=[]
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)
        images.append(io.imread(f))
# images=[io.imread(os.path.join(directory, filename) for filename in os.listdir(directory))]
plt.imshow(images[0])
# plt.show()
image=images[0]

flags=""
mytuple=(8,10)
# mySize=cv2.typing.Size(8,10)
patternSize:Size=8,10
corners=np.zeros((8,10))
retval, corners = cv2.findChessboardCorners(image, patternSize, flags)