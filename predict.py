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
from PIL import Image
import cv2

from dataloaders.data_util import Util
from dataloaders.data_reader_v2 import CapsicumDataset
from dataloaders.composed_transformer import ComposedTransformer
from train import Trainer
from constant import *


class Predictor:
    def __init__(self, model_name, output_save_dir="images"):
        self.model = self.get_model(model_name)
        self.output_save_dir = output_save_dir

    def get_model(self, model_name):
        model_path = os.path.join(model_save_dir, model_name)
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
            cv2.imwrite(os.path.join(self.output_save_dir, "pred_img1.png"), hstack)
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
        print('Image shape:', np.shape(image))

        x = image.unsqueeze(0)
        yhat = self.model(x)['out'].detach().numpy()
        print("yhat:", np.shape(yhat))

        yhat = yhat.argmax(1)
        rgb = Util.decode_segmap(yhat[0], RE_NUM_CLASS)
        print("rgb shape : ", np.shape(rgb))

        image = Image.fromarray(rgb)
        image.show(image_path)


def main():
    # the path of the image you wanna predict
    image_path = "images/synthetic_image_color_10013.png"
    # the model name
    model_name = "model_7-th_epoch.pkl"
    # the directory that the decoded output images are saved to
    output_save_dir = "images"

    predictor = Predictor(model_name=model_name, output_save_dir=output_save_dir)
    predictor.predict_from_image(image_path=image_path)
    # predictor.predict_from_loader()


if __name__ == '__main__':
    main()
