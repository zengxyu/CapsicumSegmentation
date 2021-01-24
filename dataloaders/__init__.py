# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/4
    Description :
-------------------------------------------------
"""
from dataloaders.capsicum_dataset import CapsicumDataset
from dataloaders.binary_dataset import BinaryDataset
from torch.utils.data import DataLoader


# def data_loader(args):
#     print("......Reading dataset......")
#     print()
#     train_set = BinaryDataset(root=args.data_dir, split='train', num_classes=args.num_classes,
#                               base_size=args.base_size, crop_size=args.crop_size)
#     val_set = BinaryDataset(root=args.data_dir, split='val', num_classes=args.num_classes,
#                             base_size=args.base_size, crop_size=args.crop_size)
#     train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
#                               shuffle=True, drop_last=True)
#     val_loader = DataLoader(dataset=val_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
#                             shuffle=True, drop_last=True)
#     test_loader = None
#     # batch size 的数量
#     num_image_train = len(train_set) // args.train_batch_size * args.train_batch_size
#     num_image_test = len(val_set) // args.val_batch_size * args.val_batch_size
#     print("train dataset size:{}; val dataset size:{};".format(num_image_train, num_image_test))
#     return train_loader, val_loader, test_loader

def data_loader(args):
    print("......Reading dataset......")
    print()

    Dataset = BinaryDataset if args.num_classes == 2 else CapsicumDataset

    train_set = Dataset(root=args.data_dir, split='train', num_classes=args.num_classes,
                                base_size=args.base_size, crop_size=args.crop_size)
    val_set = Dataset(root=args.data_dir, split='val', num_classes=args.num_classes,
                              base_size=args.base_size, crop_size=args.crop_size)
    train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                            shuffle=True, drop_last=True)
    test_loader = None
    # batch size 的数量
    num_image_train = len(train_set) // args.train_batch_size * args.train_batch_size
    num_image_test = len(val_set) // args.val_batch_size * args.val_batch_size
    print("train dataset size:{}; val dataset size:{};".format(num_image_train, num_image_test))
    return train_loader, val_loader, test_loader
