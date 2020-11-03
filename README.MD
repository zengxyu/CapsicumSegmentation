Here is a semantic segmentation project, 
training on the capsicum dataset,
using deeplabv3_resnet defined in Pytorch.

# The project structure:

    Capsicum Segmentation
    --assets (for dataset)
    ----images (for the images you wanna predict)
    ----log
    ----output (for the decoded predicted masks)
    ----trained_models (for saving trained models)
    --config
    --controller
    --dataloaders
    --make_data 
    --utils
    --predict.py    
    --train.py
    --test.py
    --README.MD
    --README_CN.MD
    --gitignore
    --requirements.txt (for the installation instruction)

(1) Run preprocess_bonn2019.py to make Bonn2019 to be standard capsicum-form dataset

(2) Run pepper_images_json_2_mask.py to transform annotated images from pepper_images_l515 to binary masks

(3) Run preprocess_pepper_images.py to make Bonn2019 to be standard capsicum-form dataset

(4) Run make_train_val_test_for_binary.py to make train, val, test dataset csv file, for training
    or
    Run make_train_val_test_for_binary_resume.py to make train, val, test dataset csv file, for training

(5) Run train.py to train

(6) Use following command to check the metrics ( Loss, Acc, IoU ) and intermediate predicted images
    
    tensorboard --logdir=train_log_dir --port=6006

(7) Run predict.py to predict
    change the augment "image-path" and "gt-path" to the path to your image, 
    change the augment "model-path" to the trained models


Image annotation tools, see github link:
https://github.com/uea-computer-vision/django-labeller.git


    