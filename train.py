# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/2
    Description :
-------------------------------------------------
"""

from data_reader import CapsicumDataset
from torchvision.models.segmentation import deeplabv3_resnet101
from torch import nn, optim, tensor, cuda
from torch.nn import *
from torch.utils.data import DataLoader
import numpy as np
from constant import *
import torch
import os
import datetime
import cv2
import data_reader


class Trainer:
    def __init__(self, use_gpus=True, device_ids="2", pre_trained=False):
        self.use_gpus = use_gpus
        self.model = self.define_model(pre_trained)
        self.lossfn = nn.MSELoss(reduction='mean')
        if self.use_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
            # self.model = torch.nn.DataParallel(self.model, device_ids=range(len(device_ids.split(','))))
            self.model = self.model.cuda()
            self.lossfn = self.lossfn.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def define_model(self, pre_trained):
        model = deeplabv3_resnet101(pretrained=pre_trained)
        model.classifier[-1] = Conv2d(256, NUM_CLASS, 1)
        return model

    def train_model(self, train_db=None, val_db=None, batch_size=8, epochs=100, freq_save=100, freq_val=100,
                    freq_print=5):
        self.model.train()
        train_loader = DataLoader(dataset=train_db, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_db, batch_size=batch_size, shuffle=False, drop_last=True)
        for n in range(epochs):
            print(
                "The {}-th epochs; current time : {} ".format(n, datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
            count = 0
            for x_batch, y_batch in train_loader:
                # move the data from cpu to gpu
                if self.use_gpus:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                # forward
                yhat_batch = self.model(x_batch)
                # compute the loss
                loss = self.lossfn(y_batch, yhat_batch['out'])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # update weight
                self.optimizer.step()
                # move the loss from gpu to cpu
                if self.use_gpus:
                    loss = loss.cpu().detach().numpy()
                # save model
                if count % freq_save == 0:
                    model_save_path = os.path.join(base_model_save_path,
                                                   "model_{}-th_epoch_{}-th_batch.pkl".format(n, count))
                    torch.save(self.model.state_dict(), model_save_path)
                    print("---[Save] : Save model to " + model_save_path)
                # print loss
                if count % freq_print == 0:
                    print("---[Training] : The loss of {}-th batch : {}; current time : {} ".format(count, loss,
                                                                                                    datetime.datetime.now().strftime(
                                                                                                        '%Y.%m.%d-%H:%M:%S')))

                count += 1

            # do a validation
            self.val_md(val_loader)
            # save the model after each epoch
            model_save_path = os.path.join(base_model_save_path, "model_{}-th_epoch.pkl".format(n))
            torch.save(self.model.state_dict(), model_save_path)
            print("---[Save] : Save model to " + model_save_path)

    def val_md(self, test_loader):
        count = 1
        val_loss = []
        # Do not record the gradient
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                if self.use_gpus:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                yhat_batch = self.model(x_batch)
                loss = self.lossfn(y_batch, yhat_batch['out'])
                count += 1
                val_loss.append(loss.item())
        print("---[Validation] : The mean loss : {} ".format(np.mean(val_loss)))


def train():
    # split training dataset and testing dataset
    print("......Reading dataset......")

    dataset = CapsicumDataset()
    data_size = len(dataset)
    test_size = int((1 - ratio_train_test) * data_size)
    train_size = data_size - test_size
    train_db, test_db = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("train size : {}; test size : {}.".format(train_size, test_size))

    val_size = int((1 - ratio_train_val) * train_size)
    train_size = train_size - val_size
    train_db, val_db = torch.utils.data.random_split(train_db, [train_size, val_size])
    print("train size : {}; validation size : {}.".format(train_size, val_size))

    print()
    print("......Start Training dataset......")
    model = Trainer(use_gpus=True)
    model.train_model(train_db=train_db, val_db=val_db, batch_size=16, epochs=100, freq_save=100, freq_val=150,
                      freq_print=5)
    print("......Finish training......")


if __name__ == '__main__':
    train()
