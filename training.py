import os
from functools import partial

import numpy as np
import torch

import torch.optim as optim
import statistics as s

import shutup
from numpy import savetxt

shutup.please()

from tqdm.auto import tqdm as prog

from provider.dataset_provider import get_loader
from metrics.buffered_accuarcy import bbox_accuracy

from utils.pytorchtools import (
    EarlyStopping
)

from config.configuration import (
    base_path,
    train_path_rel,
    valid_path_rel,
    device,
    num_workers,
    pin_memory,
    batch_size,
    lr
)

from model.unet_model import GlacierUNET

from segmentation_models_pytorch.losses.dice import DiceLoss

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


# The train method runs through its subset of data and puts it through the model. After that, the error will be
# calculated and back-propagated through the network just following the typical theme with some performance increasing
# tweaks of pytorch like "scaler" and "autocast"
def train(epoch, loader, loss_fn, optimizer, scaler, model):
    # Set the network to "learn" mode
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []
    running_bacc = []

    # Go through the dataset
    for batch_index, (data, target) in enumerate(loop):
        # Reset the optimizer and prepare new data
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        # Predict the data-slice
        data = model(data)

        # Prepare the target
        target = target.to(device)

        # Calculate the loss and back-propagate
        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Retrieve the actual loss value and a float value
        loss_value = loss.item()

        # Safe loss and calculate metrics (bbox accuracy, described in the git)
        running_loss.append(loss_value)

        data = torch.round(torch.sigmoid(data))
        running_bacc.append(bbox_accuracy(data, target))

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss), s.mean(running_bacc)


# The validation methid is very similar to the train method, except it does not back-propagate the error through
# the network. However, it still calculates everything else.
def valid(epoch, loader, loss_fn, model):
    # Set the network to "do not learn" mode
    model.eval()
    torch.no_grad()

    loop = prog(loader)

    running_loss = []
    running_bacc = []

    # Go through the dataset
    for batch_index, (data, target) in enumerate(loop):
        # Reset the optimizer and prepare new data (no optimizer reset needed here since no learning happens)
        data = data.to(device)

        # Predict the data-slice
        data = model(data)

        # Prepare the target
        target = target.to(device)

        # Only calculate, do not back-propagate the loss
        with torch.no_grad():
            loss = loss_fn(data, target)

        # Retrieve the actual loss value and a float value
        loss_value = loss.item()

        # Safe loss and calculate metrics (bbox accuracy, described in the git)
        running_loss.append(loss_value)

        data = torch.round(torch.sigmoid(data))
        running_bacc.append(bbox_accuracy(data, target))

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss), s.mean(running_bacc)


# The run method assembles the cycle of training and validating the model in an alternating fashion
def run(config):
    # Reset the GPU cache
    torch.cuda.empty_cache()

    # Load a new empty model with random weights
    model = GlacierUNET().to(device)

    # Initialize an optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"] if ray_tune else lr)

    # Initialize the loss, the back-propagation manager and the early stopping
    loss_fn = DiceLoss(mode="binary")
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    checkpoint = session.get_checkpoint()

    # If no ray-tune is set, we want to train a specific model to its end. The parameters set helps us to distinguish
    # the current run from other runs via the directory paths which is directly modified by those params.
    if not ray_tune:
        overall_training_loss = []
        overall_validation_loss = []
        overall_training_bacc = []
        overall_validation_bacc = []

        path = "{}_{}_{}_{}/".format(
            "results",
            str(loss_fn.__class__.__name__),
            str(optimizer.__class__.__name__),
            str(GlacierUNET.__qualname__)
        )

        if not os.path.isdir(path):
            os.mkdir(path)

    else:
        # If raytune is active, we try out parameters randomly many times. For more details, head to the git
        if checkpoint:
            checkpoint_state = checkpoint.to_dict()
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    model.to(device)

    # Load both datasets
    train_loader = get_loader(
        os.path.join(base_path, train_path_rel),
        config["batch_size"] if ray_tune else batch_size,
        num_workers,
        pin_memory
    )
    validation_loader = get_loader(
        os.path.join(base_path, valid_path_rel),
        config["batch_size"] if ray_tune else batch_size,
        num_workers,
        pin_memory
    )

    # For to endless (or max 100) epochs, train and validate in an alternating way.
    for epoch in range(0, 100):
        training_loss, training_bacc = train(
            epoch,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            model
        )

        validation_loss, validation_bacc = valid(
            epoch,
            validation_loader,
            loss_fn,
            model
        )

        # For saving, we again distinguish between if we currently try to figure out a good param set with raytune or if
        # we train out a specific parameter set.
        if ray_tune:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)

            session.report(
                {
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "training_bacc": training_bacc,
                    "validation_bacc": validation_bacc
                },
                checkpoint=checkpoint,
            )

        else:
            overall_training_loss.append(training_loss)
            overall_validation_loss.append(validation_loss)
            overall_training_bacc.append(training_bacc)
            overall_validation_bacc.append(validation_bacc)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_losses': overall_training_loss,
                'validation_losses': overall_validation_loss,
                'training_baccs': overall_training_bacc,
                'validation_baccs': overall_validation_bacc,
                'early_stopping': early_stopping
            }, path + "model_epoch" + str(epoch) + ".pt")

            model.to(device)

            metrics = np.array([
                overall_training_loss,
                overall_validation_loss,
                overall_training_bacc,
                overall_validation_bacc,
            ], dtype='object')

            # Save also a little textfile for checking everything in between that we could halt training if we see
            # something is wrong
            savetxt(path + "metrics.csv", metrics, delimiter=',',
                    header="tloss,vloss,tbacc,vbacc", fmt='%s')

        if early_stopping(validation_loss):
            print("Early stopping")
            break

        model.to(device)


def start():
    # Start the whole bs. If we use raytune, we set ranges for lr, batch-sizes, ...
    if ray_tune:
        config_space = {
            "lr": tune.loguniform(1e-6, 1e-3),
            "batch_size": tune.choice([4, 8, 16, 32])
        }
        result = tune.run(
            partial(run),
            config=config_space,
            num_samples=20,
            resources_per_trial={
                "gpu": 0.33
            }
        )

        best_trial = result.get_best_trial("validation_loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['validation_loss']}")
        print(f"Best trial final validation buffered accuracy: {best_trial.last_result['validation_bacc']}")
    else:
        run({})


# Modify this for changing between using raytune or not
ray_tune = False

if __name__ == '__main__':
    start()
