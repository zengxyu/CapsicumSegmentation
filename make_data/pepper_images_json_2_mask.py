"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/10
    Description :
                transform the annotated images, which is saved with polygon form in json files, to binary mask
                the annotated images is annotated from pepper_images_l515 dataset by hand,
                using the annotation tool, see github links :
                https://github.com/uea-computer-vision/django-labeller.git
-------------------------------------------------
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import cv2
import numpy as np
import json
from utils.common_util import load_config
from PIL import Image, ImageDraw


def read_json(json_path):
    with open(json_path, 'r') as f:
        new_dict = json.loads(f.read())
    return new_dict


def get_json_info(json_dict):
    image_filename = json_dict['image_filename']
    labels = json_dict["labels"]
    all_regison_list = []
    for label in labels:
        if label["label_class"] == "flower":
            label_regions = label["regions"]
            label_regions = label_regions[0]
            label_region_list = []
            for label_region_dict in label_regions:
                x = label_region_dict["x"]
                y = label_region_dict["y"]
                label_region_list.append([x, y])
            all_regison_list.append(label_region_list)
    return image_filename, all_regison_list


# def polygons_to_mask(img_shape, all_regison_list):
#     mask = np.zeros(img_shape, dtype=np.uint8)
#     mask = Image.fromarray(mask)
#
#     for polygons in all_regison_list:
#         xy = list(map(tuple, polygons))
#         ImageDraw.Draw(mask).polygon(xy=xy, outline=255, fill=255)
#     mask = np.array(mask)
#     mask = mask.copy()
#     mask = mask.astype(np.uint8)
#     mask[mask > 0] = 1
#     mask[mask < 0] = 1
#     return mask


def polygon2mask2(img_size, all_regison_list):
    mask = np.zeros(img_size, dtype=np.uint8)
    for polygons in all_regison_list:
        polygons = np.asarray(polygons, np.int32)
        shape = polygons.shape
        polygons = polygons.reshape(shape[0], -1, 2)
        # cv2.drawContours(mask, [polygons.astype(np.int32)], -1, color=255)
        cv2.fillConvexPoly(mask, polygons, color=255)
    mask = mask.copy()
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1
    mask[mask < 0] = 1
    mask[mask == 0] = 0
    if np.sum(mask > 1):
        print("............")
    return mask


if __name__ == '__main__':
    configs = load_config()
    root_dir = configs['root_dir']
    pepper_dir = os.path.join(root_dir, "pepper_images_l515")
    pepper_sub_dir = os.path.join(pepper_dir, "2020-07-15")
    json_file_paths = [os.path.join(pepper_sub_dir, json_file_path) for json_file_path in os.listdir(pepper_sub_dir) if
                       json_file_path.endswith(".json")]
    for json_file_path in json_file_paths:
        # print("JSON FILE PATH: ", json_file_path)
        json_dict = read_json(json_file_path)
        image_filename, all_regison_list = get_json_info(json_dict)

        ori_image = cv2.imread(os.path.join(pepper_sub_dir, image_filename))
        ori_image = ori_image.astype(np.float32)
        # print(ori_image.dtype)
        h, w, d = ori_image.shape
        mask = polygon2mask2((h, w), all_regison_list)
        cv2.imwrite(os.path.join(pepper_sub_dir, image_filename.split(".")[0] + "_label.png"), mask)
