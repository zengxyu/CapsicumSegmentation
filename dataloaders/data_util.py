# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Author :       Xiangyu Zeng
    Date：          2020/8/3
    Description :   decode the model output to color mask
-------------------------------------------------
"""
import numpy as np


class Util:
    @staticmethod
    def decode_segmap(label, nc=8):
        palette = Util.get_palette()
        r = np.zeros_like(label).astype(np.uint8)
        g = np.zeros_like(label).astype(np.uint8)
        b = np.zeros_like(label).astype(np.uint8)

        for l in range(0, nc):
            idx = label == l
            r[idx] = palette[l, 0]
            g[idx] = palette[l, 1]
            b[idx] = palette[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    @staticmethod
    def get_palette():
        palette = np.array(
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
        # 0=background
        return palette
