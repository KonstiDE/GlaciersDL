import os
import torch
import statistics as s
import numpy as np

import matplotlib.pyplot as plt

from config.configuration import (
    base_path
)


def load_graphs_from_checkpoint(model_path, epoch):
    if os.path.isfile(os.path.join(model_path, "model_epoch" + str(epoch) + ".pt")):
        checkpoint = torch.load(model_path + "model_epoch" + str(epoch) + ".pt", map_location='cpu')
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_bacc = checkpoint['training_baccs']
        overall_validation_bacc = checkpoint['validation_baccs']
        overall_training_f1 = checkpoint['training_f1s']
        overall_validation_f1 = checkpoint['validation_f1s']

        plt.figure()
        plt.plot(overall_training_loss, 'b', label="Training loss")
        plt.plot(overall_validation_loss, 'r', label="Validation loss")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_bacc, 'b', label="Training binary accuracy")
        plt.plot(overall_validation_bacc, 'o', label="Validation binary accuracy")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_f1, 'b', label="Training F1 score")
        plt.plot(overall_validation_f1, 'o', label="Validation F1 score")
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
        "results_BCEWithLogitsLoss_Adam_GlacierUNET_5e-06/"
    ), 9)
