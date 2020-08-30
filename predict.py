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
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import cv2

from dataloaders.data_util import Util
from dataloaders.capsicum import CapsicumDataset
from dataloaders.composed_transformer import ComposedTransformer
from torch.nn import *
from torchvision.models.segmentation import deeplabv3_resnet101

import matplotlib.pyplot as plt
import os
import time


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

    y_hat = predictor.predict(x)  # y_hat size = [batch_size, 1, height, width]
    y_hat_rgb = Util.decode_segmap(y_hat.argmax(1)[0], args.num_classes)
    y_hat_rgb = Image.fromarray(y_hat_rgb)
    return y_hat_rgb


def compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth_mask):
    pred_mask = ImageOps.expand(pred_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    pred_mask = np.array(pred_mask).astype(np.uint8)
    ground_truth_mask = ImageOps.expand(ground_truth_mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)
    hstack = np.hstack([pred_mask, ground_truth_mask])
    hstack = Image.fromarray(hstack)
    hstack.save(
        os.path.join(args.output, "pred_truth_compare_{}.png".format(time.strftime("%H-%M-%S", time.localtime()))))
    hstack.show("pred_truth_compare")


def predict_from_loader(self):
    """
    Predict the images loaded from the testing DataLoader
    :return:
    """
    # test dataset
    test_dataset = CapsicumDataset(root=self.args.data_dir, split="test")
    # test loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             drop_last=True)
    # iterate
    for x_batch, y_batch in test_loader:
        yhat_batch = self.model(x_batch)['out']
        rgb = Util.decode_segmap(y_batch.argmax(1)[0], self.args.num_classes)
        rgb_hat = Util.decode_segmap(yhat_batch.argmax(1)[0], self.args.num_classes)
        hstack = np.hstack([rgb, rgb_hat])
        cv2.imwrite(os.path.join(self.predicted_masks_dir, "pred_img1.png"), hstack)
        print("......Finish predicting......")
        break


def main3():
    import argparse
    parser = argparse.ArgumentParser('Pytorch Deeplabv3_resnet Predicting')
    parser.add_argument('--data-dir', type=str, default='data',
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
    image_path = "images/synthetic_image_color_1.png"
    ground_truth_path = "images/synthetic_label_class_colorscale_1.png"
    pred_mask = predict_single_image(args, image_path)
    ground_truth_mask = load_ground_truth(args, ground_truth_path)
    compare_pred_mask_and_ground_truth(args, pred_mask, ground_truth_mask)


# def main():
#     # the path of the image you wanna predict
#     image_path = "images/synthetic_image_color_1.png"
#     # the model name
#     model_path = os.path.join(model_save_dir, "model_125-th_epoch_0-th_batch.pkl")
#     # the directory that the decoded output images are saved to
#     predictor = Predictor()
#
#     predicted_mask_path = os.path.join(predicted_masks_dir,
#                                        image_path.split('/')[-1].split('.')[0] + "_mask" + ".png")
#     rgb = predictor.predict_from_image(image_path=image_path)
#     image = Image.fromarray(rgb)
#     image.save(predicted_mask_path)
#     image.show(image_path)
#     # predictor.predict_from_loader()
#
#
# def main2():
#     model_names = ["model_1-th_epoch.pkl", "model_20-th_epoch.pkl", "model_50-th_epoch.pkl", "model_80-th_epoch.pkl"]
#     model_paths = [os.path.join(model_save_dir, model_name) for model_name in model_names]
#     image_names = ["synthetic_image_color_1.png", "synthetic_image_color_100.png", "synthetic_image_color_1000.png",
#                    "synthetic_image_color_10000.png", "synthetic_image_color_10200.png"]
#     image_paths = [os.path.join("images", image_name) for image_name in image_names]
#     mask_names = ["synthetic_label_class_colorscale_1.png", "synthetic_label_class_colorscale_100.png",
#                   "synthetic_label_class_colorscale_1000.png"
#         , "synthetic_label_class_colorscale_10000.png", "synthetic_label_class_colorscale_10200.png"]
#     mask_paths = [os.path.join("images", mask_name) for mask_name in mask_names]
#
#     predicted_masks = []
#     temp_masks = [load_ground_truth(mask_path) for mask_path in mask_paths]
#     predicted_masks.append(temp_masks)
#
#     for i, model_path in enumerate(model_paths):
#         print("model_path:", model_path)
#         predictor = Predictor(model_path=model_path, predicted_masks_dir=predicted_masks_dir)
#         temp = []
#         # Use one model to predict multiple images
#         for (j, image_path) in enumerate(image_paths):
#             print("--image_path:", image_path)
#             mask = predictor.predict_from_image(image_path=image_path)
#             temp.append(mask)
#         predicted_masks.append(temp)
#     predicted_masks_display = []
#     for temp in predicted_masks:
#         hstack = np.hstack(temp)
#         predicted_masks_display.append(hstack)
#     vstack = np.vstack(predicted_masks_display)
#     mask_display = Image.fromarray(vstack)
#     mask_display.save(os.path.join(predicted_masks_dir, "mask_display.png"))
#
#     # show mask image
#
#
# def diplay_image(image_path):
#     image = Image.open(image_path)
#     plt.xlabel('images')
#     plt.ylabel('epochs')
#
#     # plt.yticks([1, 2, 3, 4, 5])
#     # plt.xticks([1, 2, 3, 4, 5])
#     plt.imshow(image)
#     plt.show()


if __name__ == '__main__':
    main3()

    # diplay_image(image_path=os.path.join(predicted_masks_dir, "mask_display.png"))
