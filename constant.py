# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/24
    Description :   some global constant variables
-------------------------------------------------
"""
import os
import datetime

# dataset root directory
# root_dir = "data"
root_dir = "E:\Tobias\Data\data\images"

# base size
base_size = 600
# crop size
crop_size = (240, 360)
# epochs for training
epochs = 1000

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
RESIZED_IMAGE_HEIGHT = 300
RESIZED_IMAGE_WIDTH = 400
# reclassified num class
RE_NUM_CLASS = 4

# The directory the model saved to
model_save_dir = "trained_models"

weight = [0.18, 0.07, 0.35, 0.4]
