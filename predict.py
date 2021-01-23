# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description :   Predict
-------------------------------------------------
"""
import numpy as np
import torch
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101

import os
import time
from PIL import Image, ImageOps

from dataloaders.capsicum_dataset import CapsicumDataset
from dataloaders.composed_transformer import ComposedTransformer
from utils import common_util
from utils.decode_util import decode_segmap


class Predictor:
    def __init__(self, device, model_path, num_classes):
        self.device = device
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[-1] = Conv2d(256, self.num_classes, 1)
        model.load_state_dict(torch.load(self.model_path, map_location='cuda:0'))
        model.eval()
        return model

    def predict(self, x):
        start_time = time.time()
        x = x.to(self.device)
        self.model = self.model.to(self.device)
        y_hat = self.model(x)['out'].cpu().detach().numpy()
        print("duration time : {}".format(time.time() - start_time))
        return y_hat


def load_ground_truth(args, mask_path):
    """
    load ground truth mask given mask
    :param args:
    :param mask_path:
    :return:
    """
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize((args.crop_size[1], args.crop_size[0]), Image.BILINEAR)

    return mask


def predict_image(args, image_path, gt_path):
    """
    predict one image, given image path
    :param args:
    :param image_path: path to image
    :return: predicted mask, dtype = uint8
    """
    predictor = Predictor(args.device, args.model_path, args.num_classes)
    # read images and gt image
    samples = {"image": Image.open(image_path).convert('RGB'), "label": Image.open(gt_path).convert('RGB')}
    # transform images and gt images
    transformed_samples = ComposedTransformer(base_size=args.base_size, crop_size=args.crop_size).transform_ts(samples)
    image, gt_image = transformed_samples["image"], transformed_samples["label"]
    # add a new dim and predict
    x = image.unsqueeze(0)
    y_hat = predictor.predict(x)  # y_hat size = [batch_size, class, height, width]
    y_hat_rgb = decode_segmap(y_hat.argmax(1)[0], args.num_classes)
    y_hat_rgb = Image.fromarray(y_hat_rgb)
    #
    gt_image = Image.fromarray(np.array(gt_image).astype(np.uint8))
    return y_hat_rgb, gt_image


def show_prediction_and_gt(args, pred_mask, gt_image, show=False):
    pred_mask = ImageOps.expand(pred_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    pred_mask = np.array(pred_mask).astype(np.uint8)

    gt_image = ImageOps.expand(gt_image, border=(0, 0, 2, 2), fill=(255, 255, 255))
    gt_image = np.array(gt_image).astype(np.uint8)
    # stack
    hstack = Image.fromarray(np.hstack([pred_mask, gt_image]))
    hstack.save(
        os.path.join(args.output,
                     "prediction_truth_comparison_{}.png".format(time.strftime("%H-%M-%S", time.localtime()))))
    if show:
        hstack.show("prediction_truth_comparison")


def get_args():
    import argparse

    configs = common_util.load_config()

    parser = argparse.ArgumentParser('Pytorch Deeplabv3_resnet Predicting')
    parser.add_argument('--base-size', type=int, default=(600, 800),
                        help='base image size')
    parser.add_argument('--crop-size', type=tuple, default=(300, 400),
                        help='crop image size')
    parser.add_argument('--num-classes', type=int, default=configs['num_classes'],
                        help='number of classes')
    parser.add_argument('--output', type=str, default='output',
                        help='where the output images are saved')
    parser.add_argument('--model-path', type=str, default='assets/trained_models/model_ep_8.pkl',
                        help='path to trained models')
    parser.add_argument('--image-path', type=str, default='assets/images/synthetic_image_color_1.png',
                        help='path to images which used for prediction')
    parser.add_argument('--gt-path', type=str, default='assets/images/synthetic_image_color_1.png',
                        help='path to ground truth')
    parser.add_argument('--use-cuda', type=bool, default=True,
                        help='whether to use cuda')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print("device :{}".format(args.device))
    return args


if __name__ == '__main__':
    args = get_args()
    # predict images
    pred_mask, gt_image = predict_image(args, args.image_path, args.image_path)
    show_prediction_and_gt(args, pred_mask, gt_image, True)
