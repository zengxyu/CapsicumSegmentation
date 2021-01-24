import os
import cv2


def read(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label2 = label.copy()
    label2[label2 == 1] = 255
    cv2.imshow("label", label2)
    cv2.waitKey(0)
    cv2.destroyWindow()


if __name__ == '__main__':
   read("E:\Tobias\Dataset_Workspace\pepper_images_l515\\2020-07-15\\frame0145_label.png")