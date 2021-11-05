import numpy as np
import cv2 as cv

img = "/home/anirban/ml/tree.png"
image = cv.imread(img)


print(image.shape)
print(image.size)
print(image.dtype)

exit(0)