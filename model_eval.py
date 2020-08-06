# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/7/26
    Description :
-------------------------------------------------
"""
import torch
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()
print(model)
filepath = "images/dog.jpg"
if not os.path.exists(filepath):
    url, filepath = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "images/dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filepath)
    except:
        urllib.request.urlretrieve(url, filepath)
# sample execution (requires torchvision)
input_image = Image.open(filepath)
print(np.shape(input_image))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# move the input and models to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

plt.imshow(r)
plt.show()
