# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/5
    Description :
-------------------------------------------------
"""

# 将所有图片背景换为白色，

# 反相

# 调节色调

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random


def main1():
    img = cv.imread('../images/synthetic_image_color_1.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mask = cv.imread("../images/synthetic_label_class_colorscale_1.png", 0)
    r, g, b = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    r[mask == 0] = random.randint(200, 255)
    g[mask == 0] = random.randint(200, 255)
    b[mask == 0] = random.randint(200, 255)
    img_back_change = np.dstack([r, g, b])
    img_back_gray = cv.cvtColor(img_back_change, cv.COLOR_RGB2GRAY)

    img_inverse = 255 - img_back_change
    img_change_hue = cv.cvtColor(img_back_change, cv.COLOR_BGR2HSV)

    plt.subplot(321), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(img_gray, cmap='gray')
    plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(323), plt.imshow(img_back_change, cmap='gray')
    plt.title('Image with inverse background'), plt.xticks([]), plt.yticks([])
    plt.subplot(324), plt.imshow(img_back_gray, cmap='gray')
    plt.title('Gray image with inverse background'), plt.xticks([]), plt.yticks([])
    plt.subplot(325), plt.imshow(img_inverse, cmap='gray')
    plt.title('Inverse Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(326), plt.imshow(img_change_hue, cmap='gray')
    plt.title('Image in hsv color space'), plt.xticks([]), plt.yticks([])
    plt.show()


def main2():
    img = cv.imread('../images/frame0000.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_inverse = 255 - img
    img_change_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_gray, cmap='gray')
    plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_inverse, cmap='gray')
    plt.title('Inverse Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_change_hue, cmap='gray')
    plt.title('Image in hsv color space'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main1()
