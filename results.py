import os

import matplotlib.pyplot as plt
import torch

import rasterio as rio
import numpy as np

import shutup
shutup.please()

from config.configuration import (
    base_path
)

from provider.dataset_provider import (
    slice_n_dice
)
from model.unet_model import GlacierUNET
from provider.front_provider import thicken_front

from PIL import Image


def generate_result(model):
    target_range = 256

    image = rio.open(os.path.join(
        base_path,
        "data/test/scenes/Crane_2002-11-18_ERS_20_3_195.png"
    )).read().squeeze(0)
    mask = rio.open(os.path.join(
        base_path,
        "data/test/masks/Crane_2002-11-18_ERS_20_3_195_front.png"
    )).read().squeeze(0)
    mask[mask == 255] = 1
    mask = thicken_front(mask, thickness=10)

    plt.imshow(mask)
    plt.show()

    mosaic_width_index = (image.shape[0] // target_range) + 1
    mosaic_height_index = (image.shape[1] // target_range) + 1

    tuples = slice_n_dice(image, mask, t=target_range)

    result_image = Image.new("L", (mosaic_height_index * target_range, mosaic_width_index * target_range))

    cw = 0
    ch = 0
    for (data, mask) in tuples:
        data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)

        data = model(data)

        if cw == mosaic_height_index:
            ch += 1
            cw = 0

        data = data.detach().numpy().squeeze(0).squeeze(0)

        result_image.paste(Image.fromarray(data), (cw * target_range, ch * target_range))

        cw += 1

    plt.imshow(result_image)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    glacier_model = GlacierUNET(in_channels=1, out_channels=1)
    glacier_model.eval()

    glacier_state_dict = torch.load(os.path.join(
        base_path,
        "results_DiceLoss_Adam_GlacierUNET_0.0001/model_epoch10.pt"
    ))["model_state_dict"]

    glacier_model.load_state_dict(glacier_state_dict)

    generate_result(model=glacier_model)
