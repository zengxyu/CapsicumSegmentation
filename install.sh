#!/bin/bash
pip3 install --user opencv-python
pip3 install --user numpy
pip3 install --user torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
echo '---------------------Finish installing'