import os
import torch

import rasterio as rio

from config.configuration import (
    base_path,
    device,
    threshold
)

from provider.dataset_provider import slice_n_dice

from PIL import Image


def generate_result(model):
    target_range = 256

    data = rio.open(os.path.join(
        base_path,
        "data/validation/scenes/Mapple_2008-11-23_PALSAR_17_5_135.png"
    )).read().squeeze(0)
    mask = rio.open(os.path.join(
        base_path,
        "data/validation/masks/Mapple_2008-11-23_PALSAR_17_5_135_front.png"
    )).read().squeeze(0)

    mosaic_width_index = (data.shape[0] // 256) + 1
    mosaic_height_index = (data.shape[1] // 256) + 1

    tuples = slice_n_dice(data, mask, t=target_range)

    result_image = Image.new("grayscale", (mosaic_width_index * target_range, mosaic_height_index * target_range))

    cw = 0
    ch = 0
    for (data, mask) in tuples:
        data = data.unsqueeze(0).unsqueeze(0).to(device)
        data = model(data)
        data[data >= threshold] = 1
        data[data < threshold] = 0

        if cw == mosaic_width_index:
            cw = 0
            ch += 1

        result_image.paste(data, (cw * target_range, ch * target_range))


if __name__ == '__main__':
    glacier_model = torch.load(os.path.join(
        base_path,
        "results_BCEWithLogitsLoss_Adam_GlacierUNET_5e-06"
    ))["state_dict"]

    generate_result(model=glacier_model)
