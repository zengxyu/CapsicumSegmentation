# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description : data reader
-------------------------------------------------
"""
from torch.utils.data import Dataset

import pickle
import numpy as np
from constant import *

from PIL import Image
from dataloaders.composed_transformer import ComposedTransformer


class CapsicumDataset(Dataset):
    def __init__(self, root="data", split="train"):
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
        # label
        label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int)

        # synthesis the label
        for i, label_path in enumerate(label_paths):
            lb = np.array(Image.open(label_path).convert('L'))
            label += lb * (i + 1)
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

        image = sample['image']
        label = self.encode_label(sample['label'])
        # sample
        return image, label

    def encode_label(self, label):
        # final_label_ont_hot
        h, w = np.shape(label)
        label_one_hot = np.zeros((h, w, NUM_CLASS + 1), dtype=np.int)
        iis, jjs = np.shape(label)
        # transform the final label into one hot, final_label_ont_hot with size [ h, w, 7+1 ]
        for i in range(iis):
            for j in range(jjs):
                try:
                    label_one_hot[i, j] = np.eye(NUM_CLASS + 1)[int(label[i, j])]
                except IndexError:
                    print(IndexError)
        # for c in range(len(iis)):
        #     label_one_hot[iis[c], jjs[c]] = np.eye(NUM_CLASS + 1)[int(label[iis[c], jjs[c]])]
        return label_one_hot.transpose((2, 0, 1))
