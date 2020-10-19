import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_seg_map_sequence(label_masks, num_classes):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, num_classes)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, num_classes):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_capsicum_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, num_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3)).astype(np.uint8)
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def get_capsicum_labels():
    return np.array(
        # background
        [(0, 0, 0),
         # leaf, blue
         (0, 0, 255),
         # capsicum, yellow
         (255, 255, 0),
         # stem, green
         (255, 0, 255),
         # stem
         (255, 100, 125),
         # stem
         (255, 255, 0),
         # stem
         (255, 0, 255),
         # stem
         (0, 255, 255)])
