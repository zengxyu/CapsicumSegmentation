#The project structure:

    Capsicum Segmentation
    --data (for dataset)
    --dataloaders
    --trained_models (for saving trained models)
    --images    (for the images you wanna predict)
    --predicted_masks (for the decoded predicted masks)
    --constant.py
    --install.sh    (for the installation instruction)
    --make_train_val_test.py    (for splitting dataset into 3 parts)
    --predict.py    
    --train.py
    --README.MD
    
The root directory of dataset is "data", defined in the constant.py,

If you wanna make the train, val, test dataset csv file, run make_train_val_test.py
If you wanna train the model, run train.py
If you wanna predict one image, 
    change the variable "image_path" to predict your image, 
    change the variable "model_name" to use another trained model

