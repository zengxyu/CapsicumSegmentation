# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/29
    Description :
-------------------------------------------------
"""
from dataloaders.data_reader import CapsicumDataset
from torchvision.models.segmentation import deeplabv3_resnet101
from torch import nn, optim
from torch.nn import *
from torch.utils.data import DataLoader
import numpy as np
from constant import *
import torch
import os
import datetime

device_ids = "2"
mapped_device_ids = range(len(device_ids.split(',')))
os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
# use_gpu = cuda.is_available()

use_gpus = True


class Model:
    def __init__(self):
        self.model = self.define_model()
        self.lossfn = nn.MSELoss(reduction='mean')
        if use_gpus:
            self.model = torch.nn.DataParallel(self.model, device_ids=mapped_device_ids)
            self.model = self.model.cuda()
            self.lossfn = self.lossfn.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def define_model(self):
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[-1] = Conv2d(256, NUM_CLASS, 1)
        return model

    def train_model(self, train_db=None, val_db=None, batch_size=8, epochs=100, freq_save=10, freq_print=5):
        self.model.train()
        train_loader = DataLoader(dataset=train_db, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_db, batch_size=batch_size, shuffle=False, drop_last=True)
        for n in range(epochs):
            print(
                "The {}-th epochs; current time : {} ".format(n, datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
            self.train_md(train_loader, n, freq_save, freq_print)
            self.val_md(val_loader)
            if (n) % freq_save == 0:
                model_save_path = os.path.join(base_model_save_path, "model_{}_th_epoch.pkl".format(n))
                torch.save(model, model_save_path)

    def train_md(self, train_loader, epoch_n, freq_save, freq_print):
        count = 1
        for x_batch, y_batch in train_loader:
            if use_gpus:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            yhat_batch = self.model(x_batch)
            loss = self.lossfn(y_batch, yhat_batch['out'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if use_gpus:
                loss = loss.cpu().detach().numpy()
            if count % freq_save == 0:
                model_save_path = os.path.join(base_model_save_path,
                                               "model_{}_th_epoch_{}_th_batch.pkl".format(epoch_n, count))
                torch.save(model, model_save_path)
            if count % freq_print == 0:
                print("---[Training] : The loss of {}-th batch : {}; current time : {} ".format(count, loss,
                                                                                                datetime.datetime.now().strftime(
                                                                                                    '%Y.%m.%d-%H:%M:%S')))
            count += 1

    def val_md(self, test_loader):
        count = 1
        val_loss = []
        # Do not record the gradient
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                if use_gpus:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                yhat_batch = self.model(x_batch)
                loss = self.lossfn(y_batch, yhat_batch['out'])
                count += 1
                val_loss.append(loss.item())
        print("---[Validation] : The mean loss : {} ".format(np.mean(val_loss)))

    # def eval(self, x, argmax=False, transform=True):
    #     self.model.eval()
    #     batch = len(x.shape) > 3
    #     x = tensor(x)
    #     if not batch:
    #         x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    #     if transform:
    #         x = self.dataset.transform(x)
    #     y = self.model(x)['out']
    #     if argmax:
    #         y = y.argmax(1)
    #     if not batch:
    #         y = y[0, :, :, :]
    #     return y

    def __getitem__(self, x):
        return self.eval(x)

    def decode_segmap(self, image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb


if __name__ == '__main__':
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
    model = Model()
    model.train_model(train_db=train_db, val_db=val_db, batch_size=4, epochs=100, freq_save=1, freq_print=100)
    print("......Finish training......")

    # img = cv.imread("images/empirical_image_color_39.png")
    # om = trained_models.eval(img)
    # rgb = trained_models.decode_segmap(om)
    # cv.imshow("rgb", rgb)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
