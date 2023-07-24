import os
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

from metrics.buffered_accuarcy import buffered_accuracy

from config.configuration import (
    base_path,
    train_path_rel,
    valid_path_rel,
    device,
    batch_size,
    num_workers,
    pin_memory
)

from model.unet_model import GlacierUNET


def train(epoch, loader, loss_fn, optimizer, scaler, model):
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []

    running_bacc0 = []
    running_bacc1 = []

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
        running_bacc0.append(buffered_accuracy(data, target, buffer_size=0))
        running_bacc1.append(buffered_accuracy(data, target, buffer_size=1))

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss), s.mean(running_bacc0), s.mean(running_bacc1)


def valid(epoch, loader, loss_fn, model):
    model.eval()
    torch.no_grad()

    loop = prog(loader)

    running_loss = []

    running_bacc0 = []
    running_bacc1 = []

    for batch_index, (data, target) in enumerate(loop):
        data = data.to(device)

        data = model(data)

        target = target.to(device)

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        running_loss.append(loss_value)
        running_bacc0.append(buffered_accuracy(data, target, buffer_size=0))
        running_bacc1.append(buffered_accuracy(data, target, buffer_size=1))

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))

    return s.mean(running_loss), s.mean(running_bacc0), s.mean(running_bacc1)


def run(num_epochs, lr, epoch_to_start_from):
    torch.cuda.empty_cache()

    model = GlacierUNET(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-04)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    epochs_done = 0

    overall_training_loss = []
    overall_validation_loss = []

    overall_training_bacc0 = []
    overall_training_bacc1 = []
    overall_validation_bacc0 = []
    overall_validation_bacc1 = []

    path = "{}_{}_{}_{}_{}/".format(
        "results",
        str(loss_fn.__class__.__name__),
        str(optimizer.__class__.__name__),
        str(GlacierUNET.__qualname__),
        lr
    )

    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.isfile(path + "model_epoch" + str(epoch_to_start_from) + ".pt") and epoch_to_start_from > 0:
        checkpoint = torch.load(path + "model_epoch" + str(epoch_to_start_from) + ".pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_bacc0 = checkpoint['training_baccs0']
        overall_validation_bacc0 = checkpoint['validation_baccs0']
        overall_training_bacc1 = checkpoint['training_baccs1']
        overall_validation_bacc1 = checkpoint['validation_baccs1']
        early_stopping = checkpoint['early_stopping']
    else:
        if epoch_to_start_from == 0:
            model.to(device)
        else:
            raise Exception("No model_epoch" + str(epoch_to_start_from) + ".pt found")

    model.to(device)

    train_loader = get_loader(os.path.join(base_path, train_path_rel), batch_size, num_workers, pin_memory)
    validation_loader = get_loader(os.path.join(base_path, valid_path_rel), batch_size, num_workers, pin_memory)

    for epoch in range(epochs_done + 1, num_epochs + 1):
        training_loss, training_bacc0, training_bacc1 = train(
            epoch,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            model
        )

        validation_loss, validation_bacc0, validation_bacc1 = valid(
            epoch,
            validation_loader,
            loss_fn,
            model
        )

        overall_training_loss.append(training_loss)
        overall_validation_loss.append(validation_loss)

        overall_training_bacc0.append(training_bacc0)
        overall_training_bacc1.append(training_bacc1)
        overall_validation_bacc0.append(validation_bacc0)
        overall_validation_bacc1.append(validation_bacc1)

        early_stopping(validation_loss, model)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': overall_training_loss,
            'validation_losses': overall_validation_loss,
            'training_baccs0': overall_training_bacc0,
            'training_baccs1': overall_training_bacc1,
            'validation_baccs0': overall_validation_bacc0,
            'validation_baccs1': overall_validation_bacc1,
            'early_stopping': early_stopping
        }, path + "model_epoch" + str(epoch) + ".pt")

        model.to(device)

        metrics = np.array([
            overall_training_loss,
            overall_validation_loss,
            overall_training_bacc0,
            overall_validation_bacc0,
            overall_training_bacc1,
            overall_validation_bacc1,
        ], dtype='object')

        savetxt(path + "metrics.csv", metrics, delimiter=',',
                header="tloss,vloss,tbacc0,vbacc0,tbacc1,vbacc1", fmt='%s')

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    run(100, lr=5e-06, epoch_to_start_from=0)
