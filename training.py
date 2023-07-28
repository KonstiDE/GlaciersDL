import os
from functools import partial

import ray
import torch
import torch.nn as nn

import torch.optim as optim
import statistics as s
import numpy as np

import shutup

shutup.please()

from numpy import savetxt
from tqdm.auto import tqdm as prog

from provider.dataset_provider import get_loader
from utils.pytorchtools import EarlyStopping

from config.configuration import (
    base_path,
    train_path_rel,
    valid_path_rel,
    device,
    num_workers,
    pin_memory,
    batch_size
)

from model.unet_model import GlacierUNET

from segmentation_models_pytorch.losses.dice import DiceLoss

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler



def train(epoch, loader, loss_fn, optimizer, scaler, model):
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []

    for batch_index, (data, target) in enumerate(loop):
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        data = model(data)

        target = target.to(device)

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_value = loss.item()

        running_loss.append(loss_value)

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss)


def valid(epoch, loader, loss_fn, model):
    model.eval()
    torch.no_grad()

    loop = prog(loader)

    running_loss = []

    for batch_index, (data, target) in enumerate(loop):
        data = data.to(device)

        data = model(data)

        target = target.to(device)

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        running_loss.append(loss_value)

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss)


def run(config):
    torch.cuda.empty_cache()

    model = GlacierUNET().to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = DiceLoss(mode="binary")
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    overall_training_loss = []
    overall_validation_loss = []

    overall_training_acc = []
    overall_validation_acc = []

    path = "{}_{}_{}_{}/".format(
        "results",
        str(loss_fn.__class__.__name__),
        str(optimizer.__class__.__name__),
        str(GlacierUNET.__qualname__)
    )

    if not os.path.isdir(path):
        os.mkdir(path)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    model.to(device)

    train_loader = get_loader(os.path.join(base_path, train_path_rel), config["batch_size"], num_workers, pin_memory)
    validation_loader = get_loader(os.path.join(base_path, valid_path_rel), config["batch_size"], num_workers, pin_memory)

    for epoch in range(start_epoch, 100):
        training_loss = train(
            epoch,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            model
        )

        validation_loss = valid(
            epoch,
            validation_loader,
            loss_fn,
            model
        )

        overall_training_loss.append(training_loss)
        overall_validation_loss.append(validation_loss)

        early_stopping(validation_loss, model)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": validation_loss},
            checkpoint=checkpoint,
        )

        model.to(device)

        metrics = np.array([
            overall_training_loss,
            overall_validation_loss,
            overall_training_acc,
            overall_validation_acc,
        ], dtype='object')

        savetxt(path + "metrics.csv", metrics, delimiter=',',
                header="tloss,vloss,tacc,vacc", fmt='%s')

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    config_space = {
        "lr": tune.loguniform(1e-1, 1e-6),
        "batch_size": tune.choice([4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(run),
        config=config_space,
        scheduler=scheduler,
        num_samples=50,
        resources_per_trial={
            "gpu": 0.25
        }
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

