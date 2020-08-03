# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/24
    Description :
-------------------------------------------------
"""
import os
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
RESIZED_IMAGE_HEIGHT = 300
RESIZED_IMAGE_WIDTH = 400
CHANNEL = 3
NUM_CLASS = 7

# base_images_dir = "E:\Tobias\Data\data\images\empirical_image_color"
# base_label_dir = "E:\Tobias\Data\data\images\empirical_label_class_grayscale"
#
# base_class_dir_format = "empirical_label_class_{}_grayscale_binary"
# base_filename_format = "empirical_label_class_{}_grayscale_{}.png"

root_dir = "data"
base_images_dir = os.path.join(root_dir, "synthetic_image_color")
base_label_dir = os.path.join(root_dir, "synthetic_label_class_grayscale")

base_class_dir_format = "synthetic_label_class_{}_grayscale_binary"
base_filename_format = "synthetic_label_class_{}_grayscale_{}.png"

base_model_save_path = "model"

ratio_train_test = 0.9
ratio_train_val = 0.9
