import os
import random

import torch

import numpy as np
import rasterio as rio
import albumentations as A

from provider.front_provider import thicken_front

from torch.utils.data import Dataset, DataLoader

import imgaug
imgaug.iarandom.seed(251199)
random.seed(251199)


# The dataset provider samples the dataset and prepares it for Pytorch to take in for training / validation. In this
# dataset, we split the data into smaller sub-samples of 256x256 pixels, to ensure a consistent size for our network.
class NrwDataSet(Dataset):
    def __init__(self, data_dir, load_amount, subs=None):

        # Directories were the images are located inside training/validation/test
        if subs is None:
            subs = ["scenes", "masks"]

        # Augmentation to apply
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.3),
        ], is_check_shapes=False)

        # This is the final list we add our tuples of (image, ground-truth)
        self.dataset = []

        # File lists
        files_data = sorted(os.listdir(os.path.join(data_dir, subs[0])))
        files_masks = sorted(os.listdir(os.path.join(data_dir, subs[1])))

        # Absolute path tuples makes followup lines much easier
        self.dataset_paths = [(
            os.path.join(data_dir, subs[0], files_data[i]),
            os.path.join(data_dir, subs[1], files_masks[i])
        ) for i in range(len(files_data))]

        c = 0
        # For each tuple: Open the file, we normalize values to ones and thicken the fronts to 10 pixels
        for path_tuple in self.dataset_paths:
            data = rio.open(path_tuple[0]).read().squeeze(0)
            mask = rio.open(path_tuple[1]).read().squeeze(0)
            mask[mask == 255] = 1
            mask = thicken_front(mask, thickness=10)

            transformed = transform(image=data, mask=mask)

            # Slice the image to smaller pieces
            tensor_slice_tuples = slice_n_dice(data, mask, t=256)
            transformed_tensor_slice_tuples = slice_n_dice(transformed["image"], transformed["mask"], t=256)

            # And add all pieces into our datasetlist
            self.dataset.extend(check_integrity(tensor_slice_tuples))
            self.dataset.extend(check_integrity(transformed_tensor_slice_tuples))

            # Debug portion for fast loading if we test things
            c += 1
            if load_amount > 0:
                print(c)

            if c == load_amount:
                break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_tuple = self.dataset[index]

        # Provide the image tuples as tensor tuples for Pytorch
        return torch.Tensor(data_tuple[0]).unsqueeze(0), \
            torch.LongTensor(data_tuple[1]).unsqueeze(0)


# Call the NrwDataSet Loader and set the desired arguments
def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, shuffle=True, load_amount=0):
    train_ds = NrwDataSet(npz_dir, load_amount)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


# Debug method if we want to get directly the data slices, not the whole loader
def get_dataset(npz_dir, load_amount=0):
    return NrwDataSet(npz_dir, load_amount)


# Slice a [b, c, h, w] tensor in smaller pieces, we also use padding if not "enough" space is
# left to make another full 256x256 image at the bottom / right edge
def slice_n_dice(data, mask, t):
    assert len(data.shape) == 2
    assert data.shape == mask.shape

    rows, cols = data.shape
    slices = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t]
            tile_mask = mask[r:r + t, c:c + t]
            if tile_data.shape != (t, t):
                padded_data = np.zeros((t, t))
                padded_mask = np.zeros((t, t))
                padded_data[:tile_data.shape[0], :tile_data.shape[1]] = tile_data
                padded_mask[:tile_mask.shape[0], :tile_mask.shape[1]] = tile_mask
                slices.append((padded_data, padded_mask))
            else:
                slices.append((tile_data, tile_mask))

    return slices


# We only include images that have something to really learn for the network.
def check_integrity(data_mask_pairs):

    valid_pairs = []

    for dm_pair in data_mask_pairs:
        if len(np.unique(dm_pair[1])) > 1:
            valid_pairs.append(dm_pair)

    return valid_pairs


if __name__ == '__main__':
    input_tensor = torch.randn(512, 513)
    input_mask = torch.randn(512, 513)

    tensor_pieces = slice_n_dice(input_tensor, input_mask, t=256)

    #for tensor_tuple in tensor_pieces:
    #    print(tensor_tuple[0].shape)
    #    print(tensor_tuple[1].shape)
