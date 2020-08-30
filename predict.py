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
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101

import os
import time
from PIL import Image, ImageOps

from dataloaders.capsicum import CapsicumDataset
from dataloaders.composed_transformer import ComposedTransformer
from util.utils import decode_segmap


class Predictor:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = deeplabv3_resnet101(pretrained=False)
        model.classifier[-1] = Conv2d(256, self.num_classes, 1)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def predict(self, x):
        y_hat = self.model(x)['out'].cpu().detach().numpy()
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


def predict_single_image(args, image_path):
    """
    predict one image, given image path
    :param args:
    :param image_path: path to image
    :return: predicted mask, dtype = uint8
    """
    cp_transformer = ComposedTransformer(base_size=args.base_size, crop_size=args.crop_size)
    predictor = Predictor(args.model_path, args.num_classes)

    image = cp_transformer.transform_ts_img(Image.open(image_path).convert('RGB'))
    x = image.unsqueeze(0)

    y_hat = predictor.predict(x)  # y_hat size = [batch_size, class, height, width]
    y_hat_rgb = decode_segmap(y_hat.argmax(1)[0], args.num_classes)
    y_hat_rgb = Image.fromarray(y_hat_rgb)
    return y_hat_rgb


def compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth_mask, show=False):
    pred_mask = ImageOps.expand(pred_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    pred_mask = np.array(pred_mask).astype(np.uint8)
    ground_truth_mask = ImageOps.expand(ground_truth_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)
    hstack = np.hstack([pred_mask, ground_truth_mask])
    hstack = Image.fromarray(hstack)
    hstack.save(
        os.path.join(args.output, "pred_truth_compare_{}.png".format(time.strftime("%H-%M-%S", time.localtime()))))
    if show:
        hstack.show("pred_truth_compare")


def predict_from_loader(args):
    """
    Predict the images loaded from the testing DataLoader
    :return:
    """
    predictor = Predictor(args.model_path, args.num_classes)
    # test dataset
    test_dataset = CapsicumDataset(root=args.data_dir, split="test")
    # test loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

    sample = test_loader.__iter__().__next__()
    x_batch, y_batch = sample['image'], sample['label']
    y_hat_batch = predictor.predict(x_batch)

    for (y_hat, y) in zip(y_hat_batch, y_batch):
        pred_mask = Image.fromarray(decode_segmap(y_hat.argmax(0), args.num_classes))
        ground_truth = Image.fromarray(decode_segmap(y.detach().cpu().numpy(), args.num_classes))
        compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth, False)
        time.sleep(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Pytorch Deeplabv3_resnet Predicting')
    parser.add_argument('--data-dir', type=str, default='E:\Tobias\Data\data\images',
                        help='where the data are placed')
    parser.add_argument('--base-size', type=int, default=600,
                        help='base image size')
    parser.add_argument('--crop-size', type=tuple, default=(300, 400),
                        help='crop image size')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='number of classes')
    parser.add_argument('--output', type=str, default='output',
                        help='where the output images are saved')
    parser.add_argument('--model-path', type=str, default='trained_models/model_20-th_epoch.pkl',
                        help='where the trained model is inputted ')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError

    # predict from loader
    # predict_from_loader(args)

    # predict images
    image_path = "images/synthetic_image_color_1.png"
    ground_truth_path = "images/synthetic_label_class_colorscale_1.png"

    pred_mask = predict_single_image(args, image_path)
    ground_truth_mask = load_ground_truth(args, ground_truth_path)
    compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth_mask, True)
