# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/2
    Description :
-------------------------------------------------
"""
import torch
import numpy as np
import cv2
from train import Trainer
from constant import *
from torchvision import transforms
from dataloaders.data_util import Util

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def eval():
    # trained_models
    model_path = "trained_models_bak/model_14-th_epoch_300-th_batch.pkl"
    network = Trainer(use_gpus=False)
    model = network.model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # image
    image_path = "data/synthetic_image_color/synthetic_image_color_1.png"
    img = cv2.imread(image_path)
    batch = len(img.shape) > 3
    # x = tensor(img)
    # if not batch:
    #     x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    x = transform(img)
    x = x.unsqueeze(0)
    y = model(x)['out']
    y = y.argmax(1)
    print("output shape : ", np.shape(y))
    # if not batch:
    #     y = y[0, :, :, :]
    rgb = Util.decode_segmap(y[0], NUM_CLASS + 1)
    print("rgb shape : ", np.shape(rgb))
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    eval()
