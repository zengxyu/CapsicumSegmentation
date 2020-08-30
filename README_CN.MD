这是个语义分割的项目，
训练在Capsicum Dataset甜椒数据集上，
用Pytorch里定义好的deeplabv3_resnet模型进行训练

# 项目结构:

    Capsicum Segmentation
    --data (for dataset)
    --dataloaders
    --trained_models (for saving trained models)
    --images    (for the images you wanna predict)
    --output (for the decoded predicted masks)
    --train_log_dir (for logging information)
    --install.sh    (for the installation instruction)
    --make_train_val_test.py    (for splitting dataset into 3 parts)
    --predict.py    
    --train.py
    --README.MD
    
(1) 运行 make_train_val_test.py 去制作 train, val, test csv 文件, 

(2) 运行 train.py 去训练模型, 

(3) 使用下列命令去查看一些度量 ( Loss, Acc, IoU ) 和 训练过程中预测出的图片
    
    tensorboard --logdir=train_log_dir --port=6006
    
(4) 运行 predict.py to 预测一张图片, 
    修改变量 "image_path" 为希想预测的图片路径, 
    修改变量 "model_name" 为希望使用的训练好的模型

