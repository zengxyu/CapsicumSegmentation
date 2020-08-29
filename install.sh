#!/bin/bash
pip3 install --user numpy
pip3 install --user tqdm
pip3 install --user pycocotools
pip3 install --user tensorboardX
pip3 install --user opencv-python
pip3 install --user sklearn
pip3 install --user pickle
pip3 install --user pillow
pip3 install --user torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
echo '---------------------Finish installing'