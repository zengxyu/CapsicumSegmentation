# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/4
    Description :
-------------------------------------------------
"""

from dataloaders.capsicum import CapsicumDataset
from torch.utils.data import DataLoader


def data_loader(args):
    print("......Reading dataset......")
    print()
    train_set = CapsicumDataset(root=args.data_dir, split='train', num_classes=args.num_classes,
                                base_size=args.base_size, crop_size=args.crop_size)
    val_set = CapsicumDataset(root=args.data_dir, split='val', num_classes=args.num_classes,
                              base_size=args.base_size, crop_size=args.crop_size)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=True)
    test_loader = None
    num_image_train = len(train_set) // args.batch_size * args.batch_size
    num_image_test = len(val_set) // args.val_batch_size * args.val_batch_size
    print("train dataset size:{}; val dataset size:{};".format(num_image_train, num_image_test))
    return train_loader, val_loader, test_loader
