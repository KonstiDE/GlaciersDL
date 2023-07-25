import os

import matplotlib.pyplot as plt
import torch

import rasterio as rio
import numpy as np

import shutup
shutup.please()

from config.configuration import (
    base_path,
    device,
    threshold
)

from provider.dataset_provider import slice_n_dice
from model.unet_model import GlacierUNET

from PIL import Image


def generate_result(model):
    target_range = 256

    image = rio.open(os.path.join(
        base_path,
        "data/validation/scenes/Mapple_2008-11-23_PALSAR_17_5_135.png"
    )).read().squeeze(0)
    mask = rio.open(os.path.join(
        base_path,
        "data/validation/masks/Mapple_2008-11-23_PALSAR_17_5_135_front.png"
    )).read().squeeze(0)

    mosaic_width_index = (image.shape[0] // 256) + 1
    mosaic_height_index = (image.shape[1] // 256) + 1

    tuples = slice_n_dice(image, mask, t=target_range)

    result_image = Image.new("1", (mosaic_width_index * target_range, mosaic_height_index * target_range))

    cw = 0
    ch = 0
    for (data, mask) in tuples:
        data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)

        data = model(data)
        data = torch.sigmoid(data)

        data[data >= threshold] = 1
        data[data < threshold] = 0

        if cw == mosaic_width_index:
            cw = 0
            ch += 1

        data = data.detach().numpy().squeeze(0).squeeze(0)

        result_image.paste(Image.fromarray(data), (cw * target_range, ch * target_range))

    plt.imshow(result_image)
    plt.show()


if __name__ == '__main__':
    glacier_model = GlacierUNET(in_channels=1, out_channels=1)

    glacier_state_dict = torch.load(os.path.join(
        base_path,
        "results_BCEWithLogitsLoss_Adam_GlacierUNET_5e-06/model_epoch62.pt"
    ))["model_state_dict"]

    glacier_model.load_state_dict(glacier_state_dict)

    generate_result(model=glacier_model)
