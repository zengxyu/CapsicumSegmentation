import os

import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import *
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from dataloaders.calculate_weights import calculate_weigths_labels
from dataloaders import data_loader
import datetime
from tqdm import tqdm

from utils.metrics import Evaluator
from utils.summaries import TensorboardSummary


class Controller:
    def __init__(self, args):
        self.args = args
        self.model = self.define_model()
        self.train_loader, self.val_loader, self.test_loader = data_loader(
            args)
        self.summary = TensorboardSummary(self.args.log_dir)
        self.writer = self.summary.create_summary()
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(args.data_dir, '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(data_dir=args.data_dir, dataloader=self.train_loader,
                                                  num_classes=self.args.num_classes)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.lossfn = nn.CrossEntropyLoss(weight=weight)

        # optimizer
        train_params = [{'params': self.model.backbone.parameters(), 'lr': args.lr},
                        {'params': self.model.classifier.parameters(), 'lr': args.lr * 10}]
        self.optimizer = optim.Adam(train_params, lr=1e-3, weight_decay=1e-5, eps=1e-08)
        self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.1)
        self.evaluator = Evaluator(self.args.num_classes)
        if self.args.resume:
            self.model.load_state_dict(torch.load(self.args.resume_path,map_location='cpu'))
            print("Load model successfully, path = ", self.args.resume_path)
        if self.args.cuda:
            self.model = self.model.cuda()
            # self.model = nn.DataParallel(self.model, device_ids=list(range(len(args.gpu_ids)))).cuda()
            self.lossfn = self.lossfn.cuda()


    def define_model(self):
        model = deeplabv3_resnet101(pretrained=self.args.pre_trained)
        model.classifier[-1] = Conv2d(256, self.args.num_classes, 1)
        # model.classifier = DeepLabHead(2048, 1)
        return model

    def training(self, epoch):
        print(
            "[ Training -- Epoch : {} ; Current time : {} ]".format(epoch, datetime.datetime.now().strftime(
                '%Y.%m.%d-%H:%M:%S')))
        self.model.train()
        train_loss = 0.0
        mean_train_loss = 0.0
        num_batch_train = len(self.train_loader)
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            x_batch, y_batch = sample['image'], sample['label']
            # move the data from cpu to gpu
            if self.args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            # forward
            yhat_batch = self.model(x_batch)
            # compute the loss
            loss = self.lossfn(yhat_batch['out'], y_batch)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # update weight
            self.optimizer.step()
            # move the loss from gpu to cpu
            train_loss += loss.item()
            mean_train_loss = train_loss / (i + 1)
            tbar.set_description("Train loss : %.3f" % mean_train_loss)
            global_step = i + num_batch_train * epoch
            self.writer.add_scalar('train/mean_loss_iteration', loss.item(), global_step)
            # visualize images ten times each epoch
            if i % 100 == 0:
                # y_batch [batch_size, width, height]
                # yhat_batch [batch_size, class, width, height]
                self.summary.visualize_image(self.writer, x_batch, y_batch, yhat_batch['out'], self.args.num_classes,
                                             global_step)
            # if i % 1000 == 0:
            #     model_save_path = os.path.join(self.args.save_model_dir, "model_ep_{}_bt_{}.pkl".format(epoch, i))
            #
            #     torch.save(self.model.state_dict(), model_save_path)
        self.writer.add_scalar('train/mean_loss_epoch', mean_train_loss, epoch)
        print('---[Loss]:{}'.format(mean_train_loss))

        return train_loss

    def validation(self, epoch):
        print(
            "[ Validation -- Epoch : {} ; Current time : {} ]".format(epoch, datetime.datetime.now().strftime(
                '%Y.%m.%d-%H:%M:%S')))
        val_loss = 0.0
        mean_val_loss = 0.0
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader)
        # Do not record the gradient
        with torch.no_grad():
            for i, samples in enumerate(tbar):
                x_batch = samples['image']
                y_batch = samples['label']
                if self.args.cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                yhat_batch = self.model(x_batch)
                loss = self.lossfn(yhat_batch['out'], y_batch)
                val_loss += loss.item()
                mean_val_loss = val_loss / (i + 1)
                tbar.set_description('Validation loss : %.3f' % mean_val_loss)

                yhat_batch = yhat_batch['out'].cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                yhat_batch = np.argmax(yhat_batch, axis=1)
                self.evaluator.add_batch(y_batch, yhat_batch)
        print("---[Loss] : {} ".format(mean_val_loss))
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/mean_loss_epoch', mean_val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)

        print(
            '---[Loss:]{}, [Acc]:{}, [Acc_class]:{}, [mIoU]:{}, [fwIoU]: {}'.format(mean_val_loss, Acc, Acc_class, mIoU,
                                                                                    FWIoU))
        return val_loss
