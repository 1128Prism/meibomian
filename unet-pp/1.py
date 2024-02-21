import os

import cv2

name = os.listdir('inputs/mg/masks/1/')
for i in range(0, len(name)):
    img = cv2.imread('inputs/mg/masks/1/' + name[i], cv2.IMREAD_GRAYSCALE) * 255
    img1 = cv2.resize(img, (896, 416))
    cv2.imshow('1',img)
    cv2.imshow('2',img1)
    cv2.waitKey(0)

    cv2.imwrite('inputs/mgm/masks/0/' + name[i], img1)
