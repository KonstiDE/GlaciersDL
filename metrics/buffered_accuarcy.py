import os.path

import numpy as np
import torch

import shutup

from config.configuration import (
    device,
    base_path
)

from PIL import Image


def buffered_accuracy(pred, target, buffersize=4):
    accs = []

    for i in range(pred.shape[0]):
        pred_raster = pred[i].squeeze(0)
        target_raster = target[i].squeeze(0)

        line_positions = torch.argwhere(target_raster == 1)

        buffer_positions = []
        for tup in line_positions:
            for i in range(1, buffersize + 1):
                buffer_positions.extend(
                    [
                        torch.tensor([[tup[0] + i, tup[1]]]).to(device),
                        torch.tensor([[tup[0] - i, tup[1]]]).to(device),
                        torch.tensor([[tup[0], tup[1] + i]]).to(device),
                        torch.tensor([[tup[0], tup[1] - i]]).to(device),
                    ]
                )

        for t in buffer_positions:
            line_positions = torch.cat((line_positions, t))

        pred_elements = pred_raster[line_positions[:, 0], line_positions[:, 1]]
        target_elements = target_raster[line_positions[:, 0], line_positions[:, 1]]

        accs.append(torch.eq(pred_elements, target_elements).sum().float().mean().item())

    return sum(accs) / len(accs)


if __name__ == '__main__':
    shutup.please()

    mask = Image.open(os.path.join(base_path, "data/validation/masks/Mapple_2018-03-07_S1_20_3_009_front.png"))
    array = np.asarray(mask.getdata())
    array = array.reshape((mask.width, mask.height))

    tensor = torch.tensor(array)
    tensor = torch.clip(tensor, 0, 1)

    A = torch.ones((4, 1, mask.width, mask.height))
    B = torch.zeros((4, 1, mask.width, mask.height))

    print(buffered_accuracy(pred=A, target=B))





