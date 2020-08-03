# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/28
    Description :
-------------------------------------------------
"""
import os
import numpy as np
import cv2 as cv
from constant import *



colors = [[255, 100, 125], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]


def make_label_image_one_hot(image_index, base_label_dir):
    final_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int)
    final_image_ont_hot = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASS), dtype=np.int)
    # synthesis the label image
    for i in range(NUM_CLASS):
        class_dir_path = os.path.join(base_label_dir, base_class_dir_format.format(i + 1))
        image_path = os.path.join(class_dir_path, base_filename_format.format(i + 1, image_index + 1))
        im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        final_image += im * (i + 1)

    # final_image_ont_hot
    iis, jjs = np.nonzero(final_image)
    # transform the final image into one hot, final_image_ont_hot with size [ h, w, 7 ]
    for c in range(len(iis)):
        final_image_ont_hot[iis[c], jjs[c]] = np.eye(NUM_CLASS)[int(final_image[iis[c], jjs[c]]) - 1]
    # transform the shape
    return final_image_ont_hot


def make_labels(base_dir_path):
    """
    :param base_dir_path: E.g: E:\Tobias\Data\data\images\empirical_label_class_grayscale
    :return:
    """
    dir_names = os.listdir(base_dir_path)
    dir_paths = []
    for dir_name in dir_names:
        if dir_name.endswith('_binary'):
            dir_paths.append(os.path.join(base_dir_path, dir_name))
    base_filename_format = "empirical_label_class_{}_grayscale_{}.png"
    image_amounts = len(os.listdir(dir_paths[0]))
    class_amounts = len(dir_paths)

    # iterate all images
    for i in range(image_amounts):
        # iterate all label_class_image, and synthesis the label image
        image_i_label_classes = []
        for j in range(class_amounts):
            file_path = os.path.join(dir_paths[j], base_filename_format.format(j + 1, i + 1))
            print(file_path)
            if os.path.exists(file_path):
                temp = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                image_i_label_classes.append(temp)
            else:
                print("This file does not exist !")
                SystemExit(0)
        # synthesis the label image
        h, w = np.shape(image_i_label_classes[0])
        final_image = np.zeros((h, w))
        final_image_ont_hot = np.zeros((h, w, class_amounts))
        for k, im in enumerate(image_i_label_classes):
            final_image += im * (k + 1)
        # final_image_ont_hot
        iis, jjs = np.nonzero(final_image)
        # transform the final image into one hot, final_image_ont_hot with size [ h, w, 7 ]
        for c in range(len(iis)):
            final_image_ont_hot[iis[c], jjs[c]] = np.eye(class_amounts)[int(final_image[iis[c], jjs[c]]) - 1]
        print(final_image_ont_hot)

# base_dir_path = "E:\Tobias\Data\data\images\empirical_label_class_grayscale"
# make_label_image_one_hot(0)
