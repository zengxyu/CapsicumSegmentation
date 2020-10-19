# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/18
    Description :
-------------------------------------------------
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils import common_util
import os
import shutil

import cv2
import numpy as np

# 输入文件夹
from tqdm import tqdm

from utils.util import check_and_create

configs = common_util.load_config()
ROOT_DIR = configs['root_dir']
INPUT_DIR_NAME = "Bonn2019_P"
OUTPUT_DIR_NAME = "Bonn2019_preprocessed"

INPUT_ROOT_DIR = os.path.join(ROOT_DIR, INPUT_DIR_NAME)  # E:\Tobias\Dataset_Workspace\Bonn2019_P
OUTPUT_ROOT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)  # E:\Tobias\Dataset_Workspace\Bonn2019_preprocessed

INPUT_SUB_DIR_NAMES = ['train', 'eval', 'val']
OUTPUT_SUB_DIR_NAMES = [OUTPUT_DIR_NAME + '_image_color',
                        OUTPUT_DIR_NAME + '_label_class_grayscale' + "/" + OUTPUT_DIR_NAME + "_label_class_2_grayscale_binary", ]

# 输出图像文件夹
# E:\Tobias\Dataset_Workspace\Bonn2019_preprocessed\Bonn2019_preprocessed_image_color
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_ROOT_DIR, OUTPUT_SUB_DIR_NAMES[0])

# E:\Tobias\Dataset_Workspace\Bonn2019_preprocessed\Bonn2019_preprocessed_label_class_grayscale
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT_DIR, OUTPUT_SUB_DIR_NAMES[1])


def combine_labels():
    check_and_create(OUTPUT_IMAGE_DIR)
    check_and_create(OUTPUT_LABEL_DIR)

    image_dir = "rgb"
    label_dir = "raw_annotation"
    count = 1
    for sub_dir in INPUT_SUB_DIR_NAMES:
        print("处理目录：", sub_dir)
        # E:\Tobias\Dataset_Workspace\Bonn2019_P\train\rgb
        in_image_dir_path = os.path.join(INPUT_ROOT_DIR, sub_dir, image_dir)

        # 合成标签
        # E:\Tobias\Dataset_Workspace\Bonn2019_P\train\raw_annotation
        in_label_dir_path = os.path.join(INPUT_ROOT_DIR, sub_dir, label_dir)
        dirs = os.listdir(in_label_dir_path)
        for frame_dir in tqdm(dirs):
            # E:\Tobias\Dataset_Workspace\Bonn2019_P\train\raw_annotation\frame_2019_10_1_8_38_33_270483
            filenames = os.listdir(os.path.join(in_label_dir_path, frame_dir))
            # E:\Tobias\Dataset_Workspace\Bonn2019_P\train\raw_annotation\frame_2019_10_1_8_38_33_270483\frame_2019_10_1_8_38_33_270483_black_0_0.png
            filepath = os.path.join(in_label_dir_path, frame_dir, filenames[0])
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_img = np.zeros(img.shape, dtype=np.int)
            for i in range(len(filenames)):
                new_img += cv2.imread(os.path.join(in_label_dir_path, frame_dir, filenames[i]), cv2.IMREAD_GRAYSCALE)
            binary_img = new_img.copy()
            binary_img = binary_img.astype(np.uint8)
            binary_img[binary_img > 0] = 1
            binary_img[binary_img < 0] = 1
            cv2.imwrite(os.path.join(OUTPUT_LABEL_DIR,
                                     OUTPUT_DIR_NAME + '_label_class_2_grayscale_' + str(count) + '.png'), binary_img)
            # 拷贝原始灯笼椒图片
            shutil.copyfile(os.path.join(in_image_dir_path, frame_dir + '.png'),
                            os.path.join(OUTPUT_IMAGE_DIR, OUTPUT_DIR_NAME + '_image_color_' + str(count) + '.png'))
            count += 1


def main():
    check_and_create(OUTPUT_ROOT_DIR)
    combine_labels()


def read(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label2 = label.copy()
    label2[label2 == 1] = 255
    cv2.imshow("label", label2)
    cv2.waitKey(0)
    cv2.destroyWindow()


if __name__ == '__main__':
    # label_path = "E:\Tobias\Dataset_Workspace\Bonn2019_preprocessed\Bonn2019_preprocessed_label_class_grayscale_binary\Bonn2019_preprocessed_label_class_2_grayscale_1.png"
    # read(label_path)
    main()
