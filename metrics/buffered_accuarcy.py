import numpy as np

import shutup


def get_bounding_box(image):
    # Get Line pixels
    positive_pixels = np.where(image == 1)

    # Determine min and max boundaries and return them
    min_row, max_row = np.min(positive_pixels[0]), np.max(positive_pixels[0])
    min_col, max_col = np.min(positive_pixels[1]), np.max(positive_pixels[1])
    return min_row, max_row, min_col, max_col


def bbox_accuracy(pred, target):
    accs = []

    # Loop over smaples in batch
    for i in range(pred.shape[0]):

        # Convert to numpy for fast computations
        pred_raster = pred[i].squeeze(0).detach().cpu().numpy()
        target_raster = target[i].squeeze(0).detach().cpu().numpy()

        # Get boundary coords
        min_row, max_row, min_col, max_col = get_bounding_box(target_raster)

        # Define area
        data_box_area = pred_raster[min_row:max_row + 1, min_col:max_col + 1]
        target_box_area = target_raster[min_row:max_row + 1, min_col:max_col + 1]

        # Check accuracy by summing equal pixels and dividing them by all bbox pixels
        correct_predictions = np.sum(data_box_area == target_box_area)
        accs.append(correct_predictions / len(data_box_area.flatten()))

    # Return accuracy of the batch
    return sum(accs) / len(accs)


if __name__ == '__main__':
    shutup.please()

    data = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0]])

    mask = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1],
                     [0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0]])

    accuracy = bbox_accuracy(data, mask)
    print(f"Bounding Box Accuracy: {accuracy:.2f}")
