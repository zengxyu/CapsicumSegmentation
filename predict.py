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
        model = Trainer(use_gpus=False).model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_from_loader(self):
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
            rgb = Util.decode_segmap(y[0], NUM_CLASS + 1)
            rgb_hat = Util.decode_segmap(yhat[0], NUM_CLASS + 1)
            print("rgb shape : ", np.shape(rgb))
            hstack = np.hstack([rgb, rgb_hat])
            cv2.imwrite(os.path.join(self.output_save_dir, "pred_img1.png"), hstack)
            print("......Finish predicting......")
            break

    def predict_from_image(self, image_path):
        """
        Given image path, predict the image segmentation by trained model
        :param image_path:
        :return:
        """
        image = Image.open(image_path).convert('RGB')
        cp_transformer = ComposedTransformer(base_size=base_size, crop_size=crop_size)
        image = cp_transformer.transform_ts_img(image)
        print('Image shape:', np.shape(image))
        x = image.unsqueeze(0)
        yhat = self.model(x)['out'].detach().numpy()
        yhat = yhat.argmax(1)
        rgb = Util.decode_segmap(yhat[0], NUM_CLASS + 1)
        print("rgb shape : ", np.shape(rgb))
        cv2.imwrite(os.path.join(self.output_save_dir, "pred_img1.png"), rgb)
        cv2.imshow("rgb", rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # predict image
    model_name = "model_4-th_epoch.pkl"
    output_save_dir = "images"
    image_path = "images/synthetic_image_color_10013.png"

    predictor = Predictor(model_name=model_name, output_save_dir=output_save_dir)
    predictor.predict_from_image(image_path=image_path)
    # predictor.predict_from_loader()


if __name__ == '__main__':
    main()
