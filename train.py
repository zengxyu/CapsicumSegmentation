# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/2
    Description :   Train
-------------------------------------------------
"""
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
import os
import datetime
from tqdm import tqdm
import shutil

from util import common_util
from util.summaries import TensorboardSummary
from util.metrics import Evaluator


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = self.define_model()
        self.train_loader, self.val_loader, self.test_loader = data_loader(
            args)
        self.summary = TensorboardSummary(self.args.train_log_dir)
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
        if self.args.cuda:
            self.model = nn.DataParallel(self.model, device_ids=list(range(len(args.gpu_ids)))).cuda()
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
            if i % (num_batch_train // 100) == 0:
                # y_batch [batch_size, width, height]
                # yhat_batch [batch_size, class, width, height]
                self.summary.visualize_image(self.writer, x_batch, y_batch, yhat_batch['out'], self.args.num_classes,
                                             global_step)

        # save the trained models after each epoch
        model_save_path = os.path.join(self.args.save_model_dir, "model_{}-th_epoch.pkl".format(epoch))
        if self.args.cuda:
            torch.save(self.model.module.state_dict(), model_save_path)
        else:
            torch.save(self.model.state_dict(), model_save_path)
        print("---[Save model] : Save model to " + self.args.save_model_dir)
        self.writer.add_scalar('train/mean_loss_epoch', mean_train_loss, epoch)
        print('---[Loss]:{}'.format(mean_train_loss))

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


def main():
    configs = common_util.load_config()
    parser = argparse.ArgumentParser("Pytorch Deeplabv3_Resnet Training")

    parser.add_argument('--train-log-dir', type=str, default='train_log_dir',
                        help='where the logs are stored')
    parser.add_argument('--data-dir', type=str, default=configs['root_dir'],
                        help='where the data are placed')
    parser.add_argument('--save-model-dir', type=str, default='trained_models',
                        help='where the model are saved')
    parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                        help='How ofter, we save the model to disk')

    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=configs['base_image_size'],
                        help='base image size')
    parser.add_argument('--crop-size', type=tuple, default=configs['crop_size'],
                        help='crop image size')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=configs['batch_size'],
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=configs['val_batch_size'],
                        metavar='N', help='input batch size for \
                                    validation (default: auto)')
    parser.add_argument('--num-classes', type=int, default=configs['num_classes'],
                        metavar='N', help='class number')

    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--pre-trained', type=bool, default=False,
                        help='whether to use pre trained deepnetv3_resnet model')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    # cuda, seed and logging
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default=configs['gpu_ids'],
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    # checking point
    parser.add_argument('--checkname', type=str, default="deeplab-resnet",
                        help='set the checkpoint name')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if os.path.exists(args.train_log_dir):
        # remove the directory first
        shutil.rmtree(args.train_log_dir)

    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)

    if not os.path.exists(args.data_dir):
        print("'{} directory is not found!'".format(args.data_dir))
        raise FileNotFoundError

    print("......Start Training......")
    trainer = Trainer(args=args)

    for n in range(args.epochs):
        print()
        # do training
        trainer.training(epoch=n)
        # do a validation
        trainer.validation(epoch=n)
        if (n + 1) % 50 == 0:
            trainer.lr_scheduler.step()

    print("......Finish training......")
    trainer.writer.close()


if __name__ == '__main__':
    main()
