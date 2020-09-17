# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/7
    Description :
-------------------------------------------------
"""
import cv2 as cv

path = "images/frame0150.jpg"
image = cv.imread(path)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imwrite(path.replace(".jpg", "_gray.jpg"), cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR))

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imwrite(path.replace(".jpg", "_hsv.jpg"), cv.cvtColor(image_hsv, cv.COLOR_BGR2RGB))
