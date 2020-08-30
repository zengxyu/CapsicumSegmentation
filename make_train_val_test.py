# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description : Split the data into 3 parts : train, val, test
-------------------------------------------------
"""

from sklearn.model_selection import train_test_split
import pickle
import os

NUM_CLASS = 7


class SplitTool(object):
    @staticmethod
    def split(root_dir, image_dir, label_dir, image_format, class_dir_format, label_format, train_test_ratio,
              train_val_ratio):
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

        # with open(os.path.join(root_dir, "val.txt"), 'rb') as f:
        #     val = pickle.load(f)
        #     f.close()
        #
        # with open(os.path.join(root_dir, "test.txt"), 'rb') as f:
        #     test = pickle.load(f)
        #     f.close()


def make_data():
    root_dir = "data"
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

    train_test_ratio = 0.9
    train_val_ratio = 0.9
    print("......Start splitting data......")
    SplitTool.split(root_dir, image_dir, label_dir, image_format, class_dir_format, label_format, train_test_ratio,
                    train_val_ratio)
    print("......Finish splitting data......")


if __name__ == '__main__':
    # SplitTool.read_data("data")
    make_data()
