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

from sklearn.model_selection import train_test_split
import pickle
import os
import random

from utils import common_util

NUM_CLASS = 7
train_test_ratio = 0.9
train_val_ratio = 0.9


class SplitTool(object):
    @staticmethod
    def split_from_emp_syb(root_dir, datasets):
        train_all = []
        val_all = []
        test_all = []
        for db in datasets:
            print("process dataset :{}".format(db))
            image_color_dir = os.path.join(root_dir, db, "{}_image_color".format(db))
            label_dir = os.path.join(root_dir, db, "{}_label_class_grayscale".format(db))
            if not os.path.exists(image_color_dir):
                print("Image base directory {} not available ! ".format(image_color_dir))
            if not os.path.exists(label_dir):
                print("Label base directory {} not available ! ".format(label_dir))
            # image paths
            im_amounts = len(os.listdir(image_color_dir))
            image_color_path = [os.path.join(image_color_dir, "{}_image_color_{}.png".format(db, i)) for i in
                                range(1, im_amounts + 1)]
            # labels paths
            lb_paths = []
            for i in range(1, im_amounts + 1):
                temp = []
                for cl in range(1, NUM_CLASS + 1):
                    class_dir = "{}_label_class_{}_grayscale_binary".format(db, cl)
                    name = "{}_label_class_{}_grayscale_{}.png".format(db, cl, i)
                    lb_path = os.path.join(label_dir, os.path.join(class_dir, name))
                    if not os.path.exists(lb_path):
                        print('Label path {} not available ! '.format(lb_path))
                        raise FileNotFoundError
                    temp.append(lb_path)
                lb_paths.append(temp)
            # mapping from image to labels
            data = []
            for i in range(im_amounts):
                data.append([image_color_path[i], lb_paths[i]])
            # split
            train, test = train_test_split(data, test_size=1 - train_test_ratio, shuffle=True)
            train, val = train_test_split(train, test_size=1 - train_val_ratio, shuffle=True)

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
            for item in train:
                print(item)
            f.close()

        with open(os.path.join(root_dir, "val.txt"), 'rb') as f:
            val = pickle.load(f)
            f.close()

        with open(os.path.join(root_dir, "test.txt"), 'rb') as f:
            test = pickle.load(f)
            f.close()


def make_data2():
    configs = common_util.load_config()
    root_dir = configs['root_dir']
    datasets = ["empirical", "synthetic"]
    SplitTool.split_from_emp_syb(root_dir, datasets)
    print("......Finish splitting data......")
    print("......Read from data file......")
    SplitTool.read_data(root_dir)



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
    # SplitTool.read_data("data")
    make_data2()
