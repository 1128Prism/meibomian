
import numpy as np


def palette(img, mask):
    gray_img = np.array(mask)
    img2 = np.zeros(img.size)
    for a in range(img2.shape[0]):
        for b in range(img2.shape[1]):
            if gray_img[a, b] == 1:
                img2[a, b, 0] = 255
                img2[a, b, 1] = 0
                img2[a, b, 2] = 0
            if gray_img[a, b] == 2:
                img2[a, b, 0] = 0
                img2[a, b, 1] = 255
                img2[a, b, 2] = 0

    mix_img = img * 0.5 + img2 * 0.5
    return mix_img
