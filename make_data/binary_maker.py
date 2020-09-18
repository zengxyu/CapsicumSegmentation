# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/9/18
    Description :
-------------------------------------------------
"""
import os

import cv2
import yaml


def test_read_label_image(filename):
    img = cv2.imread(filename)
    print(img)


if __name__ == '__main__':
    with open("../config/developer_config.yaml") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        root_dir = configs['root_dir']
        path = os.path.join(root_dir,
                            "synthetic/synthetic_label_class_grayscale/synthetic_label_class_2_grayscale_binary")
        test_read_label_image(os.path.join(path, "synthetic_label_class_2_grayscale_1.png"))
