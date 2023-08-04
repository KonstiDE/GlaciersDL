import os
import torch

import rasterio as rio
import numpy as np

import shutup
from skimage.morphology import skeletonize

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


def generate_result(model, path, subs=None):
    if subs is None:
        subs = ["scenes", "masks"]

    target_range = 256

    scenes = sorted(os.listdir(os.path.join(base_path, path, subs[0])))
    fronts = sorted(os.listdir(os.path.join(base_path, path, subs[1])))

    for index, (scene_file, front_file) in enumerate(zip(scenes, fronts)):
        image = rio.open(os.path.join(
            base_path,
            path,
            subs[0],
            scene_file
        )).read().squeeze(0)
        mask = rio.open(os.path.join(
            base_path,
            path,
            subs[1],
            front_file
        )).read().squeeze(0)
        mask[mask == 255] = 1
        mask = thicken_front(mask, thickness=10)

        mosaic_width_index = (image.shape[0] // target_range) + 1
        mosaic_height_index = (image.shape[1] // target_range) + 1

        tuples = slice_n_dice(image, mask, t=target_range)

        result_image = Image.new("L", (mosaic_height_index * target_range, mosaic_width_index * target_range))

        cw = 0
        ch = 0
        for (data, mask) in tuples:
            data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)

            data = model(data)
            data = torch.sigmoid(data)

            if cw == mosaic_height_index:
                ch += 1
                cw = 0

            data = data.detach().numpy().squeeze(0).squeeze(0)

            result_image.paste(Image.fromarray(data), (cw * target_range, ch * target_range))

            cw += 1

        result_image = result_image.crop((0, 0, image.shape[1], image.shape[0]))

        matrix = np.array(result_image.getdata())
        matrix = matrix.reshape(image.shape)
        matrix = skeletonize(matrix)

        img = Image.fromarray(matrix)
        img.save(os.path.join(base_path, "results", scene_file + "_pred.png"))

        print("Finished " + scene_file + " (" + str((index / (len(scenes) - 1)) * 100) + "%)")


def skeletonize_longest_line(matrix):
    skeleton = skeletonize(matrix)
    return skeleton.astype(int)


if __name__ == '__main__':
    glacier_model = GlacierUNET(in_channels=1, out_channels=1)
    glacier_model.eval()

    glacier_state_dict = torch.load(os.path.join(
        base_path,
        "results_DiceLoss_Adam_GlacierUNET/model_epoch6.pt"
    ))["model_state_dict"]
    glacier_model.load_state_dict(glacier_state_dict)

    generate_result(model=glacier_model, path="data/test")


