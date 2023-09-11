import os
import torch
import statistics as s
import numpy as np

import matplotlib.pyplot as plt

from config.configuration import (
    base_path
)


# With this method we extract the saved metrics from our .pt file format and generate easy receptable graphs for an
# arbitrary epoch
def load_graphs_from_checkpoint(model_path, epoch):
    if os.path.isfile(os.path.join(model_path, "model_epoch" + str(epoch) + ".pt")):
        checkpoint = torch.load(model_path + "model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_bacc = checkpoint['training_baccs']
        overall_validation_bacc = checkpoint['validation_baccs']

        plt.figure()
        plt.plot(overall_training_loss, 'b', label="Training loss")
        plt.plot(overall_validation_loss, 'r', label="Validation loss")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_bacc, 'b', label="Training bbox accuracy")
        plt.plot(overall_validation_bacc, 'r', label="Validation bbox accuracy")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

    else:
        print("No model found within {} and epoch {}".format(
            model_path,
            str(epoch)
        ))


if __name__ == '__main__':
    load_graphs_from_checkpoint(os.path.join(
        base_path,
        "results_DiceLoss_Adam_GlacierUNET_3x3_dil1_ASPPconv_max512/"
    ), 11)
