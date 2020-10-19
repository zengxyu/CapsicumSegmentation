# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/2
    Description :   Train
-------------------------------------------------
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse
import torch
import numpy as np

from utils import common_util
from controller.controller import Controller


def main():
    configs = common_util.load_config()
    parser = argparse.ArgumentParser("Pytorch Deeplabv3_Resnet Training")

    parser.add_argument('--data-dir', type=str, default=configs['root_dir'],
                        help='where the data are placed')
    parser.add_argument('--log-dir', type=str, default='assets/log',
                        help='where the logs are stored')
    parser.add_argument('--save-model-dir', type=str, default='assets/trained_models',
                        help='where the model are saved')
    parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                        help='How ofter, we save the model to disk')
    # cuda, num workers
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default=configs['gpu_ids'],
                        help='use which gpu to train, must be a \
                                comma-separated list of integers only (default=0)')
    parser.add_argument('--num_workers', type=int, default=configs['num_workers'], metavar='N',
                        help='dataloader threads')

    # resume
    parser.add_argument('--resume', type=bool, default=configs['resume'],
                        help='whether to resume')
    parser.add_argument('--resume_path', type=str, default=configs['resume_path'],
                        help='path to resume model')
    # image size
    parser.add_argument('--base-size', type=tuple, default=(720, 1280),
                        help='base image size')
    parser.add_argument('--crop-size', type=tuple, default=(300, 400),
                        help='crop image size')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--train-batch-size', type=int, default=configs['train_batch_size'],
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=configs['val_batch_size'],
                        metavar='N', help='input batch size for \
                                    validation (default: auto)')
    parser.add_argument('--num-classes', type=int, default=configs['num_classes'],
                        metavar='N', help='class number')

    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--pre-trained', type=bool, default=True,
                        help='whether to use pre trained deepnetv3_resnet model')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    args = parser.parse_args()
    # args.crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.cuda:
        print("Use NVIDIA GPU to train")
        try:
            print("GPU Ids: ", args.gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            torch.cuda.set_device(args.gpu_ids[0])

        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    else:
        print("Use CPU to train")

    # check directory
    if not os.path.exists(args.data_dir):
        print("'{} directory is not found!'".format(args.data_dir))
        raise FileNotFoundError

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    if args.resume and not os.path.exists(args.resume_path):
        print("'{} file is not found!'".format(args.resume_path))
        raise FileNotFoundError

    print("......Start Training......")
    controller = Controller(args=args)
    best_val_loss = np.inf
    for n in range(args.epochs):
        print()
        # do training
        train_loss = controller.training(epoch=n)
        # do a validation
        val_loss = controller.validation(epoch=n)
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            # save the trained models after each epoch
            model_save_path = os.path.join(args.save_model_dir, "model_ep_{}.pkl".format(n))
            # if args.cuda:
            #     torch.save(controller.model.module.state_dict(), model_save_path)
            # else:
            torch.save(controller.model.state_dict(), model_save_path)
            print("---[Save model] : Save model to " + args.save_model_dir)

        if (n + 1) % 50 == 0:
            controller.lr_scheduler.step()

    print("......Finish training......")
    controller.writer.close()


if __name__ == '__main__':
    main()
