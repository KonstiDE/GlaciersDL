import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio as rio
import torchvision.transforms.functional as tf

import shutup

from config.configuration import (
    base_path
)


def buffered_accuracy(pred, target, buffer_size=1):
    accs = []

    for i in range(pred.shape[0]):
        pred_raster = pred[i].squeeze(0)
        target_raster = target[i].squeeze(0)

        plt.imshow(pred_raster.detach().cpu().numpy())
        plt.show()

        plt.imshow(target_raster.detach().cpu().numpy())
        plt.show()

        line_positions = torch.argwhere(target_raster == 1)

        if buffer_size == 0:
            pred_elements = pred_raster[line_positions[:, 0], line_positions[:, 1]]
            target_elements = target_raster[line_positions[:, 0], line_positions[:, 1]]

            return (torch.eq(pred_elements, target_elements).sum() / len(line_positions)).item()

        hits = 0
        pixels = 0

        done = set()

        for pos in line_positions:
            row, col = pos[0].item(), pos[1].item()

            for i in range(-buffer_size, buffer_size + 1):
                if (row + i, col) not in done:
                    done.add((row + i, col))
                    if 0 <= row + i < pred_raster.shape[0]:
                        pixels += 1
                        if torch.equal(pred_raster[row + i, col], target_raster[row + i, col]):
                            hits += 1

                if (row, col + i) not in done:
                    done.add((row, col + i))
                    if 0 <= col + i < pred_raster.shape[1]:
                        pixels += 1
                        if torch.equal(pred_raster[row, col + i], target_raster[row, col + i]):
                            hits += 1

        accs.append(hits / pixels)

    return sum(accs) / len(accs)


if __name__ == '__main__':
    shutup.please()

    mask = rio.open(os.path.join(base_path, "data/validation/masks/Mapple_2008-11-23_PALSAR_17_5_135_front.png")).read()

    array = np.stack((mask, mask, mask, mask))

    tensor = torch.tensor(array).cuda()
    tensor = torch.clip(tensor, 0, 1)

    B = torch.ones((4, 1, tensor.shape[2], tensor.shape[3])).cuda()

    print(B.shape)
    print(tensor.shape)

    pred_update = buffered_accuracy(pred=B, target=tensor)
    print(pred_update)





