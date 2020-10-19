# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/7
    Description :
-------------------------------------------------
"""
import cv2 as cv
import os

# path = "assets/images/frame0150.jpg"
# image = cv.imread(path)
# image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cv.imwrite(path.replace(".jpg", "_gray.jpg"), cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR))
#
# image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# cv.imwrite(path.replace(".jpg", "_hsv.jpg"), cv.cvtColor(image_hsv, cv.COLOR_BGR2RGB))
# path = "E:\Tobias\Dataset_Workspace\empirical\empirical_label_class_grayscale\empirical_label_class_2_grayscale_binary\empirical_label_class_2_grayscale_1.png"
path = "E:\Tobias\Dataset_Workspace\synthetic\synthetic_label_class_grayscale\synthetic_label_class_2_grayscale_binary\synthetic_label_class_2_grayscale_1.png"
if __name__ == '__main__':
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img[img == 1] = 255
    cv.imshow("test", img)
    cv.waitKey(0)
    cv.destroyWindow()
