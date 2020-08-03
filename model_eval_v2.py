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
import data_reader
from torch import tensor
from torchvision.models.segmentation import deeplabv3_resnet101
from train import Trainer
from constant import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def eval():
    # model
    model_path = "model/model_12-th_epoch_200-th_batch.pkl"
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
    rgb = decode_segmap(y[0], NUM_CLASS)
    print("rgb shape : ", np.shape(rgb))
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decode_segmap(image, nc=21):
    # 0=background
    colors = np.array([(0, 0, 0), (255, 100, 125), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                       (0, 255, 255)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = colors[l, 0]
        g[idx] = colors[l, 1]
        b[idx] = colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


if __name__ == '__main__':
    eval()
