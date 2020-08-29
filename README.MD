Here is a semantic segmentation project, 
training on the dataset capsicum dataset,
using deeplabv3_resnet defined in Pytorch.

#The project structure:

    Capsicum Segmentation
    --data (for dataset)
    --dataloaders
    --trained_models (for saving trained models)
    --images    (for the images you wanna predict)
    --predicted_masks (for the decoded predicted masks)
    --train_log_dir (for logging information)
    --constant.py
    --install.sh    (for the installation instruction)
    --make_train_val_test.py    (for splitting dataset into 3 parts)
    --predict.py    
    --train.py
    --README.MD
    

#Run make_train_val_test.py to make train, val, test dataset csv file, 
#Run train.py to train the model, 
#Run predict.py to predict an image, 
    ##change the variable "image_path" to predict your image, 
    ##change the variable "model_name" to use another trained model

