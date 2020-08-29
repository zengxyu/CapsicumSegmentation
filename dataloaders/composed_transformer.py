# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/4
    Description :  transformers for training dataset, validation dataset, and testing dataset
-------------------------------------------------
"""
from torchvision import transforms
from dataloaders import custom_transforms as ct


class ComposedTransformer:
    def __init__(self, base_size, crop_size):
        self.base_size = base_size
        self.crop_size = crop_size

    def transform_tr(self, sample):
        """
        composed transformers for training dataset
        :param sample: {'image': image, 'label': label}
        :return:
        """
        img = sample['image']
        img = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(img)
        sample = {'image': img, 'label': sample['label']}
        composed_transforms = transforms.Compose([
            ct.RandomHorizontalFlip(),
            ct.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            ct.RandomGaussianBlur(),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        """
       composed transformers for validation dataset
       :param sample: {'image': image, 'label': label}
       :return:
       """
        composed_transforms = transforms.Compose([
            ct.FixScaleCrop(crop_size=self.crop_size),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):
        """
           composed transformers for testing dataset
           :param sample: {'image': image, 'label': label}
           :return:
           """
        composed_transforms = transforms.Compose([
            ct.FixedResize(size=self.crop_size),
            ct.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ct.ToTensor()])

        return composed_transforms(sample)

    def transform_ts_img(self, image):
        """
          composed transformers for testing image, for predicting an image
          :param image: only image, no mask,
          :return:
          """
        composed_transforms = transforms.Compose([
            ct.FixedResizeImage(size=self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        return composed_transforms(image)
