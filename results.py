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


# This method is similar to what the provider and the training script does, however, combined. We split the image again
# into patches of 256x256 and put it through the now trained network (loaded in the __main__ method)
def generate_result(model, path, subs=None):
    # See the dataset_provider for more details in the following lines
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

        # Again edit and thicken the image line to be acceptable for the network
        mask[mask == 255] = 1
        mask = thicken_front(mask, thickness=10)

        # Calculate indices for that we know how to split and later merge the image together correctly
        mosaic_width_index = (image.shape[0] // target_range) + 1
        mosaic_height_index = (image.shape[1] // target_range) + 1

        # Finally slice
        tuples = slice_n_dice(image, mask, t=target_range)

        # Make a new empty image where we will store the predictions at the right positions
        result_image = Image.new("L", (mosaic_height_index * target_range, mosaic_width_index * target_range))

        cw = 0
        ch = 0
        for (data, mask) in tuples:
            # Now predict a slice and apply sigmoid for probabilities for each pixel. Also, a threshold could be used
            # additionally
            data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)

            data = model(data)
            data = torch.sigmoid(data)

            if cw == mosaic_height_index:
                ch += 1
                cw = 0

            # Some more squeezing as PIL images cannot display arrays with batch-size and channel dimensions
            data = data.detach().numpy().squeeze(0).squeeze(0)

            # Put the prediction into our resulting image
            result_image.paste(Image.fromarray(data), (cw * target_range, ch * target_range))

            cw += 1

        # Final cropping
        result_image = result_image.crop((0, 0, image.shape[1], image.shape[0]))

        # Skeletanization to retrieve and exact prediction. Otherwise, also the predictions in the timeseries would
        # overlap
        matrix = np.array(result_image.getdata())
        matrix = matrix.reshape(image.shape)
        matrix = skeletonize(matrix)

        # Save the image
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
        "results_DiceLoss_Adam_GlacierUNET_3x3_dil1_ASPPconv_max512/model_epoch6.pt"
    ))["model_state_dict"]
    glacier_model.load_state_dict(glacier_state_dict)

    generate_result(model=glacier_model, path="data/test")


