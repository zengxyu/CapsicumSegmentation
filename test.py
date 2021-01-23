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
import numpy as np

path = "E:\Tobias\Dataset_Workspace\Bonn2019_P\\train\\raw_annotation\\frame_2019_10_1_8_38_33_270483"

image_names = os.listdir(path)
synthetic_images = None
for image_name in image_names:
    image_path = os.path.join(path, image_name)
    img = cv.imread(image_path)
    img_cvt = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if synthetic_images is None:
        synthetic_images = img_cvt
    else:
        synthetic_images += img_cvt

h, w = synthetic_images.shape
print(h, w)
synthetic_images = cv.resize(synthetic_images, (int(w * 0.5), int(h * 0.5)))
h, w = synthetic_images.shape
print(h, w)
cv.imshow("images", synthetic_images)
cv.waitKey(0)
cv.destroyAllWindows()
