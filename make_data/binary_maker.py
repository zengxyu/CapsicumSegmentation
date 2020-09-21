# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/18
    Description :
-------------------------------------------------
"""
import os
import shutil

import cv2
import numpy as np

# 输入文件夹
from tqdm import tqdm

INPUT_ROOT_DIR = "H:/original/bell_pepper/Bonn2019_P"
SUB_DIRS = ['train', 'eval', 'val']
# 输出文件夹
OUTPUT_DIR_NAME = "Bonn2019"
OUTPUT_ROOT_DIR = "G:/data/pepper_seg/binary_class/" + OUTPUT_DIR_NAME
OUTPUT_IMAGE_DIR = OUTPUT_ROOT_DIR + '/' + OUTPUT_DIR_NAME + '_image_color'
OUTPUT_LABEL_SEEN_DIR = OUTPUT_ROOT_DIR + '/' + OUTPUT_DIR_NAME + '_label_class_grayscale'
OUTPUT_LABEL_DIR = OUTPUT_ROOT_DIR + '/' + OUTPUT_DIR_NAME + '_label_class_grayscale_binary'


def check_and_create(path):
    paths = path.replace("\\", '/').split('/')
    dir_path = None
    for p in paths:
        if dir_path is None:
            dir_path = p
        else:
            dir_path = os.path.join(dir_path, p)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def combine_labels():
    check_and_create(OUTPUT_IMAGE_DIR)
    check_and_create(OUTPUT_LABEL_SEEN_DIR)
    check_and_create(OUTPUT_LABEL_DIR)

    image_dir = "rgb"
    label_dir = "raw_annotation"
    count = 1
    for sub_dir in SUB_DIRS:
        print("处理目录：", sub_dir)

        in_image_dir_path = os.path.join(INPUT_ROOT_DIR, sub_dir, image_dir)

        # 合成标签
        in_label_dir_path = os.path.join(INPUT_ROOT_DIR, sub_dir, label_dir)
        dirs = os.listdir(in_label_dir_path)
        for dir in tqdm(dirs):
            filenames = os.listdir(os.path.join(in_label_dir_path, dir))
            p = os.path.join(in_label_dir_path, dir, filenames[0])
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            new_img = np.zeros(img.shape, dtype=np.int)
            for i in range(len(filenames)):
                new_img += cv2.imread(os.path.join(in_label_dir_path, dir, filenames[i]), cv2.IMREAD_GRAYSCALE)
            new_img, binary_img = reformat_img(new_img)
            cv2.imwrite(os.path.join(OUTPUT_LABEL_SEEN_DIR,
                                     OUTPUT_DIR_NAME + '_label_class_2_grayscale_' + str(count) + '.png'), new_img)
            cv2.imwrite(os.path.join(OUTPUT_LABEL_DIR,
                                     OUTPUT_DIR_NAME + '_label_class_2_grayscale_' + str(count) + '.png'), binary_img)
            # 拷贝原始灯笼椒图片
            shutil.copyfile(os.path.join(in_image_dir_path, dir + '.png'),
                            os.path.join(OUTPUT_IMAGE_DIR, OUTPUT_DIR_NAME + '_image_color_' + str(count) + '.png'))
            count += 1


def reformat_img(img):
    h, w = img.shape
    binary_img = np.zeros(img.shape, np.int8)
    for i in range(h):
        for j in range(w):
            if img[i, j] >= 255:
                img[i, j] = 255
                binary_img[i, j] = 1
    return img, binary_img


if __name__ == '__main__':
    check_and_create(OUTPUT_ROOT_DIR)
    combine_labels()
