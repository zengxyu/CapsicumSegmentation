# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description :
-------------------------------------------------
"""
from torch.utils.data import DataLoader, Dataset, random_split


class CapsicumDataset(Dataset):
    def __init__(self):
