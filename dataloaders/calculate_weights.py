import os
from tqdm import tqdm
import numpy as np

import random


def calculate_weigths_labels(data_dir, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        x_batch, y_batch = sample['image'], sample['label']
        y_batch = y_batch.detach().cpu().numpy()
        mask = (y_batch >= 0) & (y_batch < num_classes)
        labels = y_batch[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(data_dir, "_classes_weights.npy")
    np.save(classes_weights_path, ret)

    return ret
