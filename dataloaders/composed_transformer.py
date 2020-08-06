# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/4
    Description :
-------------------------------------------------
"""
from torchvision import transforms
from dataloaders import custom_transforms as ct


class ComposedTransformer:
    def __init__(self, base_size, crop_size):
        self.base_size = base_size
        self.crop_size = crop_size

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            ct.RandomHorizontalFlip(),
            ct.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=0),
            ct.RandomGaussianBlur(),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            ct.FixScaleCrop(crop_size=self.crop_size),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            ct.FixedResize(size=self.crop_size),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_ts_img(self, image):
        composed_transforms = transforms.Compose([
            ct.FixedResizeImage(size=self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        return composed_transforms(image)
