# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/10/12
    Description :
-------------------------------------------------
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import colorsys
from PIL import Image
from tqdm import tqdm

from utils import common_util


def make_augmented_dataset(root_dir, sub_dir):
    dir_read = os.path.join(root_dir, sub_dir, sub_dir + "_image_color")
    if not os.path.exists(dir_read):
        print("输入文件夹不存在!!!")
        return
    # 1. 灰度图 2. 反向图 3. HSV颜色图
    transforms = ["image_gray", "image_inverse", "image_hsv"]
    # 创建输出目录
    out_dirs = [os.path.join(root_dir, sub_dir, sub_dir + "_" + t) for t in transforms]
    for d in out_dirs:
        os.mkdir(d) if not os.path.exists(d) else None
    for filename in tqdm(os.listdir(dir_read)):
        # 原图
        img_path = os.path.join(dir_read, filename)
        img = cv.imread(img_path)
        # 2.背景变换后的灰度图
        img_back_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite(os.path.join(out_dirs[0], filename), cv.cvtColor(img_back_gray, cv.COLOR_GRAY2BGR))
        # 3.背景变换后的反相图
        img_inverse = 255 - img
        cv.imwrite(os.path.join(out_dirs[1], filename), img_inverse)
        # 4.背景变换后的HSV颜色空间图
        img_change_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        cv.imwrite(os.path.join(out_dirs[2], filename), cv.cvtColor(img_change_hue, cv.COLOR_BGR2RGB))
    # plt.subplot(321), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(322), plt.imshow(img_gray, cmap='gray')
    # plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(323), plt.imshow(img_back_change, cmap='gray')
    # plt.title('Image with inverse background'), plt.xticks([]), plt.yticks([])
    # plt.subplot(324), plt.imshow(img_back_gray, cmap='gray')
    # plt.title('Gray image with inverse background'), plt.xticks([]), plt.yticks([])
    # plt.subplot(325), plt.imshow(img_inverse, cmap='gray')
    # plt.title('Inverse Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(326), plt.imshow(img_change_hue, cmap='gray')
    # plt.title('Image in hsv color space'), plt.xticks([]), plt.yticks([])
    # plt.show()


if __name__ == '__main__':
    configs = common_util.load_config()
    root_dir = configs['root_dir']
    sub_dirs = ['Bonn2019_preprocessed']
    for dir in sub_dirs:
        print("处理目录", dir)
        make_augmented_dataset(root_dir, dir)
