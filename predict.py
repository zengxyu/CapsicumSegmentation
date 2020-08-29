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
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import cv2

from dataloaders.data_util import Util
from dataloaders.capsicum import CapsicumDataset
from dataloaders.composed_transformer import ComposedTransformer
from train import Trainer
from constant import *
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, model_path, predicted_masks_dir="images"):
        self.model = self.get_model(model_path)
        self.predicted_masks_dir = predicted_masks_dir

    def get_model(self, model_path):
        model = Trainer(use_gpus=False, use_balanced_weights=False).model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_from_loader(self):
        """
        Predict the images loaded from the testing DataLoader
        :return:
        """
        # test dataset
        test_dataset = CapsicumDataset(root=root_dir, split="test")
        test_dataset_size = len(test_dataset)
        print("Test dataset size : ", test_dataset_size)
        # test loader
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                 drop_last=True)
        # iterate
        for x_batch, y_batch in test_loader:
            print("x_batch shape : ", np.shape(y_batch))
            print("y_batch shape : ", np.shape(y_batch))
            y = y_batch.argmax(1)
            yhat_batch = self.model(x_batch)['out']
            print("yhat_batch shape : ", np.shape(yhat_batch))
            yhat = yhat_batch.argmax(1)
            rgb = Util.decode_segmap(y[0], RE_NUM_CLASS)
            rgb_hat = Util.decode_segmap(yhat[0], RE_NUM_CLASS)
            print("rgb shape : ", np.shape(rgb))
            hstack = np.hstack([rgb, rgb_hat])
            cv2.imwrite(os.path.join(self.predicted_masks_dir, "pred_img1.png"), hstack)
            print("......Finish predicting......")
            break

    def predict_from_image(self, image_path):
        """
        Given image path, predict the mask by trained model
        :param image_path:
        :return:
        """
        image = Image.open(image_path).convert('RGB')
        cp_transformer = ComposedTransformer(base_size=base_size, crop_size=crop_size)
        image = cp_transformer.transform_ts_img(image)
        # util('Image shape:', np.shape(image))

        x = image.unsqueeze(0)
        yhat = self.model(x)['out'].detach().numpy()
        # util("yhat:", np.shape(yhat))

        yhat = yhat.argmax(1)
        rgb = Util.decode_segmap(yhat[0], RE_NUM_CLASS)
        # util("rgb shape : ", np.shape(rgb))
        rgb = ImageOps.expand(Image.fromarray(rgb), border=(0, 0, 2, 2), fill=(255, 255, 255))
        rgb = np.array(rgb).astype(np.uint8)
        return rgb



def load_mask(mask_path):
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize((crop_size[1], crop_size[0]), Image.BILINEAR)
    mask = ImageOps.expand(mask, border=(0, 0, 2, 2), fill=(255, 255, 255))
    mask = np.array(mask).astype(np.uint8)
    return mask


def main():
    # the path of the image you wanna predict
    image_path = "images/synthetic_image_color_1.png"
    # the model name
    model_path = os.path.join(model_save_dir, "model_125-th_epoch_0-th_batch.pkl")
    # the directory that the decoded output images are saved to
    predictor = Predictor(model_path=model_path, predicted_masks_dir=predicted_masks_dir)

    predicted_mask_path = os.path.join(predicted_masks_dir,
                                       image_path.split('/')[-1].split('.')[0] + "_mask" + ".png")
    rgb = predictor.predict_from_image(image_path=image_path)
    image = Image.fromarray(rgb)
    image.save(predicted_mask_path)
    image.show(image_path)
    # predictor.predict_from_loader()


def main2():
    model_names = ["model_1-th_epoch.pkl", "model_20-th_epoch.pkl", "model_50-th_epoch.pkl", "model_80-th_epoch.pkl"]
    model_paths = [os.path.join(model_save_dir, model_name) for model_name in model_names]
    image_names = ["synthetic_image_color_1.png", "synthetic_image_color_100.png", "synthetic_image_color_1000.png",
                   "synthetic_image_color_10000.png", "synthetic_image_color_10200.png"]
    image_paths = [os.path.join("images", image_name) for image_name in image_names]
    mask_names = ["synthetic_label_class_colorscale_1.png", "synthetic_label_class_colorscale_100.png",
                  "synthetic_label_class_colorscale_1000.png"
        , "synthetic_label_class_colorscale_10000.png", "synthetic_label_class_colorscale_10200.png"]
    mask_paths = [os.path.join("images", mask_name) for mask_name in mask_names]

    predicted_masks = []
    temp_masks = [load_mask(mask_path) for mask_path in mask_paths]
    predicted_masks.append(temp_masks)

    for i, model_path in enumerate(model_paths):
        print("model_path:", model_path)
        predictor = Predictor(model_path=model_path, predicted_masks_dir=predicted_masks_dir)
        temp = []
        # Use one model to predict multiple images
        for (j, image_path) in enumerate(image_paths):
            print("--image_path:", image_path)
            mask = predictor.predict_from_image(image_path=image_path)
            temp.append(mask)
        predicted_masks.append(temp)
    predicted_masks_display = []
    for temp in predicted_masks:
        hstack = np.hstack(temp)
        predicted_masks_display.append(hstack)
    vstack = np.vstack(predicted_masks_display)
    mask_display = Image.fromarray(vstack)
    mask_display.save(os.path.join(predicted_masks_dir, "mask_display.png"))

    # show mask image


def diplay_image(image_path):
    image = Image.open(image_path)
    plt.xlabel('images')
    plt.ylabel('epochs')

    # plt.yticks([1, 2, 3, 4, 5])
    # plt.xticks([1, 2, 3, 4, 5])
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main2()

    # diplay_image(image_path=os.path.join(predicted_masks_dir, "mask_display.png"))
