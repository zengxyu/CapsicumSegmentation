# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/26
    Description :
-------------------------------------------------
"""
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import torch
from torch import nn


class ImageDataset(Dataset):

    def __init__(self, xstream=None, ystream=None, switchdim=True, tofloat=True, normalize=True, customtransform=None,
                 device=None):
        if xstream is not None:
            self.x = torch.stack([x for x in xstream], dim=0)
            self.y = torch.stack([y for y in ystream], dim=0)
        else:
            self.x = torch.empty(0, 0, 0, 0)
            self.y = torch.empty(0, 0, 0, 0)
        self.switchdim = switchdim
        self.tofloat = tofloat
        self.normalize = normalize
        self.customtransform = customtransform
        self.device = device
        self.x = self.transform(self.x)
        self.y = self.to_one_hot(self.y.permute(0, 3, 1, 2))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.customtransform is not None:
            x, y = self.customtransform(x, y)  # create wrappers of transforms with __call__(self, *images)
        return (x, y)

    def __len__(self):
        return self.x.shape[0]

    def transform(self, x):
        # If input is only three dimensional, temporarily add a fourth dimension
        batch = len(x.shape) > 3
        if not batch:
            x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        # Switch dimensions from [samples, height, width, channels] to [samples, channels, height, width]
        if self.switchdim:
            x = x.permute(0, 3, 1, 2)
        # Convert from [0,255] uint8 to [0,1) float
        if self.tofloat:
            x = x.float() / 255.0
        # Apply the default input normalization for pytorch
        if self.normalize:
            x = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        # Put data on proper device
        if self.device is not None:
            x = x.to(self.device)
        # If input was only three dimensional, remove fourth dimension now
        if not batch:
            x = x[0, :, :, :]
        return x

    def to_one_hot(x, nclasses):
        shape = x.shape
        x = torch.eye(nclasses)[:, x.view(-1)]
        x = x.T
        x = x.view(shape[0], shape[2], shape[3], nclasses)
        x = x.permute(0, 3, 1, 2)
        return x

# Define custom transformations here (random crop, random rotation, etc)
