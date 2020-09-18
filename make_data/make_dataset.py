# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/5
    Description :
-------------------------------------------------
"""
import os

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import colorsys
from PIL import Image
from tqdm import tqdm

from util import common_util


def make_augmented_dataset(root_dir, sub_dir):
    dir_read = os.path.join(root_dir, sub_dir, sub_dir + "_image_color")
    dir_mask_read = os.path.join(root_dir, sub_dir, sub_dir + "_label_class_colorscale")
    if not os.path.exists(dir_read):
        print("输入文件夹不存在!!!")
        return
    # 1. 先把背景转为随机的浅色，2. 转换后背景的灰度图， 3. 转换背景后反相图，4. 转换背景后的HSV颜色图片
    transforms = ["image_ibg", "image_ibg_gray", "image_ibg_inverse", "image_ibg_hsv"]
    # 创建输出目录
    out_dirs = [os.path.join(root_dir, sub_dir, sub_dir + "_" + t) for t in transforms]
    for d in out_dirs:
        os.mkdir(d) if not os.path.exists(d) else None
    for filename in tqdm(os.listdir(dir_read)):
        # 原图
        img_path = os.path.join(dir_read, filename)
        img = cv.imread(img_path)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # 灰度图
        # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask_path = img_path.replace("image_color", "label_class_colorscale")
        # 1.原图背景变成浅色,
        mask = cv.imread(mask_path, 0)
        r, g, b = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
        r[mask == 0] = random.randint(200, 255)
        g[mask == 0] = random.randint(200, 255)
        b[mask == 0] = random.randint(200, 255)
        img_back_change = np.dstack([r, g, b])
        cv.imwrite(os.path.join(out_dirs[0], filename), img_back_change)
        # 2.背景变换后的灰度图
        img_back_gray = cv.cvtColor(img_back_change, cv.COLOR_BGR2GRAY)
        cv.imwrite(os.path.join(out_dirs[1], filename), cv.cvtColor(img_back_gray, cv.COLOR_GRAY2BGR))
        # 3.背景变换后的反相图
        img_inverse = 255 - img_back_change
        cv.imwrite(os.path.join(out_dirs[2], filename), img_inverse)
        # 4.背景变换后的HSV颜色空间图
        img_change_hue = cv.cvtColor(img_back_change, cv.COLOR_BGR2HSV)
        cv.imwrite(os.path.join(out_dirs[3], filename), cv.cvtColor(img_change_hue, cv.COLOR_BGR2RGB))
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


def change_hue(img):
    # 目标色值
    target_hue = 180

    # 读入图片，转化为 RGB 色值
    image = Image.fromarray(img)

    # 将 RGB 色值分离
    image.load()
    r, g, b = image.split()
    result_r, result_g, result_b = [], [], []
    # 依次对每个像素点进行处理
    for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
        # 转为 HSV 色值
        h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
        # 转回 RGB 色系
        rgb = colorsys.hsv_to_rgb(target_hue, s, v)
        pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
        # 每个像素点结果保存
        result_r.append(pixel_r)
        result_g.append(pixel_g)
        result_b.append(pixel_b)

    r.putdata(result_r)
    g.putdata(result_g)
    b.putdata(result_b)

    # 合并图片
    image = Image.merge('RGB', (r, g, b))
    # 输出图片
    return np.array(image)


if __name__ == '__main__':
    configs = common_util.load_config()
    root_dir = configs['root_dir']
    sub_dirs = ['synthetic', 'empirical']
    for dir in sub_dirs:
        print("处理目录", dir)
        make_augmented_dataset(root_dir, dir)
    # change_hue()
