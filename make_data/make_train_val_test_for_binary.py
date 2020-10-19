# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description : Split the data into 3 parts : train, val, test
-------------------------------------------------
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import pickle
import random

from utils import common_util
import cv2 as cv

NUM_CLASS = 2
split_ratio = [0.8, 0.1, 0.1]
# datasets_images = {
#     "synthetic": ["synthetic_image_color", "synthetic_image_ibg", "synthetic_image_ibg_gray", "synthetic_image_ibg_hsv",
#                   "synthetic_image_ibg_inverse"],
#     "empirical": ["empirical_image_color", "empirical_image_ibg", "empirical_image_ibg_gray", "empirical_image_ibg_hsv",
#                   "empirical_image_ibg_inverse"],
#     "Bonn2019_preprocessed": ["Bonn2019_preprocessed_image_color", "Bonn2019_preprocessed_image_gray",
#                               "Bonn2019_preprocessed_image_hsv", "Bonn2019_preprocessed_image_inverse"]
# }

datasets_images = {
    "synthetic": ["synthetic_image_color"],
    "empirical": ["empirical_image_color"],
    "Bonn2019_preprocessed": ["Bonn2019_preprocessed_image_color"]
}
datasets_annotations = {
    "synthetic": "synthetic_label_class_grayscale",
    "empirical": "empirical_label_class_grayscale",
    "Bonn2019_preprocessed": "Bonn2019_preprocessed_label_class_grayscale"
}

base_num = 0


class SplitTool(object):
    @staticmethod
    def add_label(label_dir, label_paths, db, i):
        name = "{}_label_class_{}_grayscale_{}.png".format(db, 2, i)
        lb_path = os.path.join(label_dir, name)
        if not os.path.exists(lb_path):
            print('Label path {} not available ! '.format(lb_path))
            raise FileNotFoundError
        label_paths.append(lb_path)

    # TODO 小数据集要写多遍
    @staticmethod
    def split_from_emp_syb(root_dir):
        global base_num
        train_all = []
        val_all = []
        test_all = []
        for db in datasets_images.keys():
            for img_dir_name in datasets_images[db]:
                # 'E:\\Tobias\\Dataset_Workspace\\synthetic\\synthetic_image_color'
                image_dir = os.path.join(root_dir, db, img_dir_name)
                # 'E:\\Tobias\\Dataset_Workspace\\synthetic\\synthetic_label_class_grayscale\\synthetic_label_class_2_grayscale_binary'
                label_dir = os.path.join(root_dir, db, datasets_annotations[db],
                                         "{}_label_class_2_grayscale_binary".format(db))
                if not os.path.exists(image_dir):
                    print("Image base directory {} not available ! ".format(image_dir))
                if not os.path.exists(label_dir):
                    print("Label base directory {} not available ! ".format(label_dir))

                # image paths
                im_amounts = len(os.listdir(image_dir))
                image_path = [os.path.join(image_dir, "{}_image_color_{}.png".format(db, i)) for i in
                              range(1, im_amounts + 1)]
                if base_num == 0:
                    base_num = im_amounts
                ratio = int(base_num / im_amounts)
                # labels paths
                lb_paths = []
                for i in range(1, im_amounts + 1):
                    SplitTool.add_label(label_dir, lb_paths, db, i)

                # mapping from image to labels
                data = []
                for k in range(ratio):
                    for i in range(im_amounts):
                        data.append([image_path[i], lb_paths[i]])
                print("data len:", len(data))
                # split
                # train, test = train_test_split(data, test_size=1 - train_test_ratio, shuffle=False)
                # train, val = train_test_split(train, test_size=1 - train_val_ratio, shuffle=False)
                random.shuffle(data)
                train_len = int(len(data) * split_ratio[0])
                val_len = int(len(data) * split_ratio[1])
                test_len = len(data) - train_len - val_len
                train = data[:train_len]
                val = data[train_len:train_len + val_len]
                test = data[train_len + val_len:]
                train_all.extend(train)
                val_all.extend(val)
                test_all.extend(test)
        random.shuffle(train_all)
        random.shuffle(val_all)
        random.shuffle(test_all)
        print(len(train_all))
        print(".......Write to txt file......")
        # write to train.txt, val.txt, test.txt
        with open(os.path.join(root_dir, "train.txt"), 'wb') as f:
            pickle.dump(train_all, f)
            f.close()
            print(".......Train dataset made !")
        with open(os.path.join(root_dir, "val.txt"), 'wb') as f:
            pickle.dump(val_all, f)
            f.close()
            print(".......Val dataset made !")
        with open(os.path.join(root_dir, "test.txt"), 'wb') as f:
            pickle.dump(test_all, f)
            f.close()
            print(".......Test dataset made !")

    @staticmethod
    def split(root_dir, image_dir, label_dir, image_format, class_dir_format, label_format):
        image_dir = os.path.join(root_dir, image_dir)
        label_dir = os.path.join(root_dir, label_dir)
        if not os.path.exists(image_dir):
            print("Image base directory {} not available ! ".format(image_dir))
        if not os.path.exists(label_dir):
            print("Label base directory {} not available ! ".format(label_dir))
        # image paths
        im_amounts = len(os.listdir(image_dir))
        im_paths = [os.path.join(image_dir, image_format.format(i)) for i in
                    range(1, im_amounts + 1)]
        # labels paths
        lb_paths = []
        for i in range(1, im_amounts + 1):
            temp = []
            for cl in range(1, NUM_CLASS + 1):
                class_dir = class_dir_format.format(cl)
                name = label_format.format(cl, i)
                lb_path = os.path.join(label_dir, os.path.join(class_dir, name))
                if not os.path.exists(lb_path):
                    print('Label path {} not available ! '.format(lb_path))
                    raise FileNotFoundError
                temp.append(lb_path)
            lb_paths.append(temp)
        # mapping from image to labels
        data = []
        for i in range(im_amounts):
            data.append([im_paths[i], lb_paths[i]])

        # split
        train, test = train_test_split(data, test_size=1 - train_test_ratio, shuffle=False)
        train, val = train_test_split(train, test_size=1 - train_val_ratio, shuffle=False)
        print(".......Write to txt file......")
        # write to train.txt, val.txt, test.txt
        with open(os.path.join(root_dir, "train.txt"), 'wb') as f:
            pickle.dump(train, f)
            f.close()
            print(".......Train dataset made !")
        with open(os.path.join(root_dir, "val.txt"), 'wb') as f:
            pickle.dump(val, f)
            f.close()
            print(".......Val dataset made !")
        with open(os.path.join(root_dir, "test.txt"), 'wb') as f:
            pickle.dump(test, f)
            f.close()
            print(".......Test dataset made !")

    @staticmethod
    def read_data(root_dir):
        with open(os.path.join(root_dir, "train.txt"), 'rb') as f:
            train = pickle.load(f)
            item = random.choice(train)
            im_path = item[0]
            lb_path = item[1]
            print("im_path:", im_path)
            print("lb_path:", lb_path)
            im = cv.imread(im_path)
            lb = cv.imread(lb_path)
            lb[lb == 1] = 225
            cv.imshow("im", im)
            cv.imshow("lb", lb)
            cv.waitKey(0)
            cv.destroyWindow()

        # with open(os.path.join(root_dir, "val.txt"), 'rb') as f:
        #     val = pickle.load(f)
        #     f.close()
        #
        # with open(os.path.join(root_dir, "test.txt"), 'rb') as f:
        #     test = pickle.load(f)
        #     f.close()


def make_data2():
    configs = common_util.load_config()
    root_dir = configs['root_dir']
    SplitTool.split_from_emp_syb(root_dir)
    print("......Finish splitting data......")


def make_data():
    root_dir = "../data"
    image_dir = "synthetic_image_color"
    label_dir = "synthetic_label_class_grayscale"
    image_format = "synthetic_image_color_{}.png"
    class_dir_format = "synthetic_label_class_{}_grayscale_binary"
    label_format = "synthetic_label_class_{}_grayscale_{}.png"

    # root_dir = "E:\Tobias\Data\data\images"
    # image_dir = "empirical_image_color"
    # label_dir = "empirical_label_class_grayscale"
    # image_format = "empirical_image_color_{}.png"
    # class_dir_format = "empirical_label_class_{}_grayscale_binary"
    # label_format = "empirical_label_class_{}_grayscale_{}.png"

    print("......Start splitting data......")
    SplitTool.split(root_dir, image_dir, label_dir, image_format, class_dir_format, label_format)
    print("......Finish splitting data......")


if __name__ == '__main__':
    configs = common_util.load_config()
    root_dir = configs['root_dir']
    SplitTool.read_data(root_dir)
    # make_data2()
