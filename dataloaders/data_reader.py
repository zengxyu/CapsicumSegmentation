# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/29
    Description :
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/24
    Description :
-------------------------------------------------
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from constant import *

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH)),
    # transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class CapsicumDataset(Dataset):
    def __init__(self):
        self.transform = transform
        self.color_map = {}

    def __len__(self):
        return len(os.listdir(image_dir_base))

    def __getitem__(self, idx):
        img_name = os.listdir(image_dir_base)[idx]
        imgA = cv2.imread(os.path.join(image_dir_base, img_name))
        # imgA = cv2.resize(imgA, (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH))
        imgA = self.transform(imgA)  # 一转成向量后，imgA通道就变成(C,H,W)
        # print(imgA)
        # imgA = cv2.resize(imgA, (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH,3),cv2.INTER_AREA)
        # print(imgA.shape)
        imgB = self.get_label_image_one_hot(image_index=idx)
        imgB = imgB.transpose(2, 0, 1)  # imgB不经过transform处理，所以要手动把(H,W,C)转成(C,H,W)
        imgB = torch.FloatTensor(imgB)
        # imgB = cv2.resize(imgB, (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH),cv2.INTER_AREA)
        return imgA, imgB

    def get_label_image_one_hot(self, image_index):
        final_image = np.zeros((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH), dtype=np.int)
        final_image_ont_hot = np.zeros((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, NUM_CLASS), dtype=np.int)
        rows = np.arange(start=0, stop=IMAGE_HEIGHT, step=2)
        cols = np.arange(start=0, stop=IMAGE_WIDTH, step=2)
        # synthesis the label image
        for i in range(NUM_CLASS):
            class_dir_path = os.path.join(label_dir_base, class_dir_format.format(i + 1))
            image_path = os.path.join(class_dir_path, filename_format.format(i + 1, image_index + 1))
            if not os.path.exists(image_path):
                print(" The image path does not exist ! ")
            im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # resize the image
            im = im[rows][:, cols]
            # synthesis
            final_image += im * (i + 1)

        # final_image_ont_hot
        iis, jjs = np.nonzero(final_image)
        # transform the final image into one hot, final_image_ont_hot with size [ h, w, 7 ]
        for c in range(len(iis)):
            final_image_ont_hot[iis[c], jjs[c]] = np.eye(NUM_CLASS)[int(final_image[iis[c], jjs[c]]) - 1]
        # transform the shape
        return final_image_ont_hot

# bag = CapsicumDataset()
#
# train_size = int(0.9 * len(bag))  # 整个训练集中，百分之90为训练集
# test_size = len(bag) - train_size
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])  # 划分训练集和测试集
#
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
#
# if __name__ == '__main__':
#     for train_batch in train_dataloader:
#         print(train_batch)
#
#     for test_batch in test_dataloader:
#         print(test_batch)
