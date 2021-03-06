# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description : data reader
-------------------------------------------------
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import os

from dataloaders.composed_transformer import ComposedTransformer


class CapsicumDataset(Dataset):
    def __init__(self, root="data", split="train", base_size=600, crop_size=(300, 400)):
        self.num_classes = 4
        self.image_height = None
        self.image_width = None
        self.root = root
        self.split = split
        self.data_file = None
        self.data = []
        if self.split == "train":
            self.data_file = open(os.path.join(root, "train.txt"), 'rb')
        elif self.split == "val":
            self.data_file = open(os.path.join(root, "val.txt"), 'rb')
        elif self.split == "test":
            self.data_file = open(os.path.join(root, "test.txt"), 'rb')
        else:
            print("You are allowed to choose data source from 'train','val','test' ! ")
            raise NotImplementedError
        self.data = pickle.load(self.data_file)
        self.cp_transformer = ComposedTransformer(base_size=base_size, crop_size=crop_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label_paths = self.data[index]
        # image
        image = Image.open(image_path).convert('RGB')
        if self.image_height is None or self.image_width is None:
            self.image_width, self.image_height = image.size
        # label
        label = np.zeros((self.image_height, self.image_width), dtype=np.int)
        # synthesis the label
        for i, label_path in enumerate(label_paths):
            lb = np.array(Image.open(label_path).convert('L'))
            # 0-background, 1-leaf, 2-capsicum, 3,4,5,6,7-different kinds of stems
            if i < self.num_classes - 1:
                label += lb * (i + 1)
            else:
                label += lb * (self.num_classes - 1)
        label = Image.fromarray(label.astype(np.uint8))
        # sample
        sample = {'image': image, 'label': label}

        # transform
        if self.split == 'train':
            sample = self.cp_transformer.transform_tr(sample)
        elif self.split == 'val':
            sample = self.cp_transformer.transform_val(sample)
        elif self.split == 'test':
            sample = self.cp_transformer.transform_ts(sample)

        sample = {'image': sample['image'].type(torch.FloatTensor), 'label': sample['label'].type(torch.LongTensor)}
        # sample
        return sample
