# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/2
    Description :   Train
-------------------------------------------------
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import *
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from dataloaders.data_reader_v2 import CapsicumDataset
from dataloaders.calculate_weights import calculate_weigths_labels
from constant import *


class Trainer:
    def __init__(self, use_gpus=True, use_balanced_weights=True, device_ids="2", train_db=None, val_db=None,
                 batch_size=8,
                 model_save_dir="trained_models", pre_trained=False):
        self.use_gpus = use_gpus
        self.model = self.define_model(pre_trained)
        self.train_loader = DataLoader(dataset=train_db, batch_size=batch_size, shuffle=True,
                                       drop_last=True) if train_db is not None else None
        self.val_loader = DataLoader(dataset=val_db, batch_size=batch_size, shuffle=False,
                                     drop_last=True) if val_db is not None else None

        if use_balanced_weights:
            classes_weights_path = os.path.join(root_dir, '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(dataloader=self.train_loader, num_classes=RE_NUM_CLASS)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.lossfn = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9,
        #                                  weight_decay=1e-5, lr=1e-4, nesterov=False)

        self.model_save_dir = model_save_dir
        if self.use_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
            self.model = self.model.cuda()
            self.lossfn = self.lossfn.cuda()

    def define_model(self, pre_trained):
        model = deeplabv3_resnet101(pretrained=pre_trained, aux_loss=True)
        model.classifier[-1] = Conv2d(256, RE_NUM_CLASS, 1)
        # model.classifier = DeepLabHead(2048, 1)
        return model

    def training(self, epoch, freq_save=100, freq_val=100, freq_print=5):
        self.model.train()
        print(
            "[ Epoch : {} ; Current time : {} ]".format(epoch, datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
        count = 0
        for x_batch, y_batch in self.train_loader:
            # move the data from cpu to gpu
            if self.use_gpus:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            # forward
            yhat_batch = self.model(x_batch)
            # compute the loss
            loss1 = self.lossfn(yhat_batch['out'], y_batch)
            loss2 = self.lossfn(yhat_batch['aux'], y_batch)
            loss = loss1 + 0.3 * loss2
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # update weight
            self.optimizer.step()
            # move the loss from gpu to cpu
            if self.use_gpus:
                loss = loss.cpu().detach().numpy()
            # save trained models
            if count % freq_save == 0:
                model_save_path = os.path.join(self.model_save_dir,
                                               "model_{}-th_epoch_{}-th_batch.pkl".format(epoch, count))
                torch.save(self.model.state_dict(), model_save_path)
                print("---[Save] : Save model to " + model_save_path)
            # print loss
            if count % freq_print == 0:
                print("---[Training] : Loss of {}-th batch : {}; Current time : {} ".format(count, loss,
                                                                                            datetime.datetime.now().strftime(
                                                                                                '%Y.%m.%d-%H:%M:%S')))

            count += 1
            # save the trained models after each epoch
        model_save_path = os.path.join(self.model_save_dir, "model_{}-th_epoch.pkl".format(epoch))
        torch.save(self.model.state_dict(), model_save_path)
        print("---[Save] : Save model to " + self.model_save_dir)

    def validation(self, epoch):
        val_loss = []
        print(
            "[ Epoch : {} ; Current time : {} ]".format(epoch, datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
        # Do not record the gradient
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                if self.use_gpus:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                yhat_batch = self.model(x_batch)
                loss1 = self.lossfn(yhat_batch['out'], y_batch)
                loss2 = self.lossfn(yhat_batch['aux'], y_batch)
                loss = loss1 + 0.3 * loss2
                val_loss.append(loss.item())

        print("---[Validation] : Mean loss : {} ".format(np.mean(val_loss)))


def main():
    # split training dataset and testing dataset
    print("......Reading dataset......")
    train_dataset = CapsicumDataset(root=root_dir, split="train")
    train_dataset_size = len(train_dataset)
    test_dataset = CapsicumDataset(root=root_dir, split="test")
    test_dataset_size = len(test_dataset)
    val_dataset = CapsicumDataset(root=root_dir, split="val")
    val_dataset_size = len(val_dataset)
    print("train size : {}; test size : {}; val size : {} .".format(train_dataset_size, test_dataset_size,
                                                                    val_dataset_size))

    print()
    print("......Start Training dataset......")
    trainer = Trainer(use_gpus=True, use_balanced_weights=True, device_ids='3', train_db=train_dataset,
                      val_db=val_dataset, batch_size=16,
                      model_save_dir="trained_models",
                      pre_trained=False)

    for n in range(epochs):
        # do training
        trainer.training(epoch=n, freq_save=500, freq_val=150,
                         freq_print=50)
        # do a validation
        trainer.validation(epoch=n)

    print("......Finish training......")


if __name__ == '__main__':
    main()
