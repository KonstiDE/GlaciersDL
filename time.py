import os

import matplotlib.pyplot as plt

import rasterio as rio
import numpy as np

import cv2

import shutup
from rasterio.transform import from_origin

shutup.please()

from config.configuration import (
    base_path
)

from PIL import Image


def create_color_gradient(n_classes):
    cmap = plt.get_cmap('jet')  # You can choose different colormaps
    colors = [cmap(i) for i in np.linspace(0, 1, n_classes)]
    return colors


def colorize_masks_stacked(stacked_masks, colors):
    h, w, n_layers = stacked_masks.shape
    colored_stack = np.zeros((h, w, 3), dtype=np.uint8)

    colors = (np.array(colors)[:, :3] * 255).astype(np.uint8)

    for i in range(n_layers):
        mask = stacked_masks[:, :, i]
        mask = mask[:, :, np.newaxis]
        colored_stack += (mask * colors[i])

    return colored_stack


def time(path):
    files = sorted(os.listdir(os.path.join(base_path, path)))

    stack = []
    underlying = None

    for index, (file) in enumerate(files):
        if file.__contains__("_pred.png"):
            pred = rio.open(os.path.join(
                base_path,
                path,
                file
            )).read().squeeze(0)

            pred[pred == 1] = index + 1

            stack.append(pred)
        else:
            underlying = rio.open(os.path.join(
                base_path,
                path,
                file
            )).read().squeeze(0)

    stack = np.stack(stack, axis=-1)
    stack = np.argmax(stack, axis=-1)

    print(np.unique(stack))
    print(stack.shape)

    transform = from_origin(472137, 5015782, 0.5, 0.5)

    lines = rio.open('results.tif', 'w', driver='GTiff',
                     height=stack.shape[0], width=stack.shape[1],
                     count=1, dtype=str(stack.dtype),
                     crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                     transform=transform)

    base_layer = rio.open('base.tif', 'w', driver='GTiff',
                          height=underlying.shape[0], width=underlying.shape[1],
                          count=1, dtype=str(underlying.dtype),
                          crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                          transform=transform)

    lines.write(stack, 1)
    lines.close()

    base_layer.write(underlying, 1)
    base_layer.close()


if __name__ == '__main__':
    time(path="time")
