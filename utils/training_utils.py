from time import time

import torch
import numpy as np

from torch import nn, no_grad
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models import Network, RealNVP
from models.real_nvp.utils import clip_grad_norm
from models.real_nvp.real_nvp_loss import RealNVPLoss
from utils import utils
from utils.data_utils import ImageDataset
from tqdm import tqdm


def train(net: Network, data_loader: DataLoader, optimizer: Optimizer):
    net.train()

    for inputs, targets in data_loader:
        optimizer.zero_grad()

        prediction = net(inputs)
        loss = F.binary_cross_entropy(prediction, targets)

        loss.backward()
        optimizer.step()

        if net.convex:
            net.step()

    return float(loss)


def train_realnvp(net: RealNVP, data_loader: DataLoader, optimizer: Optimizer,
                  loss_fn=RealNVPLoss(), device='cpu'):
    net.train()
    total_loss = []
    with tqdm(total=len(data_loader.dataset)) as progress:
        for inputs, _ in data_loader:
            inputs.to(device, dtype=float)
            optimizer.zero_grad()

            z, sldj = net(inputs)
            loss = loss_fn(z, sldj)

            loss.backward()
            optimizer.step()

            total_loss.append(float(loss))
            progress.set_postfix(loss=float(loss))
            progress.update(len(inputs))

    return sum(total_loss) / len(total_loss)


def test_realnvp(net: nn.Module, data_loader: DataLoader, loss_fn=RealNVPLoss(), device='cpu'):
    net.eval()
    loss = 0.

    with no_grad():
        for inputs, _ in data_loader:
            inputs.to(device)
            z, sldj = net(inputs)
            loss += loss_fn(z, sldj)

    return loss/len(data_loader), 1


def test(net: nn.Module, data_loader: DataLoader):
    net.eval()
    total_loss = 0.
    total_correct = 0
    with no_grad():
        for inputs, targets in data_loader:
            predictions = net(inputs)

            loss = F.binary_cross_entropy(predictions, targets)

            total_loss += float(loss)
            total_correct += torch.round(predictions).eq(targets).sum().item()

    return total_loss / len(data_loader.dataset), total_correct / (len(data_loader.dataset))


def full_training(model: Network,
                  train_loader: DataLoader, test_loader: DataLoader,
                  optimizer: Optimizer, lr_scheduler: _LRScheduler,
                  snapshots_folder: str = r".snapshots", start_epoch: int = 0,
                  model_name: str = ""):

    utils.setup_checkpoint(snapshots_folder)
    utils.setup_log(snapshots_folder, model_name)

    print(f"Starting training from epoch: {start_epoch}")
    for epoch in range(start_epoch, 50):
        start_time = time()

        if model_name == "real_nvp":
            train_loss = train_realnvp(model, train_loader, optimizer)
            test_loss, test_acc = test_realnvp(model, test_loader)
        else:
            train_loss = train(model, train_loader, optimizer)
            test_loss, test_acc = test(model, test_loader)

        if lr_scheduler:
            lr_scheduler.step(test_loss)

        utils.log(epoch, start_time, train_loss, test_loss, test_acc)
        utils.checkpoint(model, optimizer, epoch, test_loss, model_name)
    print(
        f"The training is finished successfully! The model is saved to ./{snapshots_folder}")


def inference(network: Network, path_to_model: str, path_to_image: str):

    checkpoint = torch.load(path_to_model)
    network.load_state_dict(checkpoint["network"])

    image = ImageDataset.preprocess_image(path_to_image)
    features = ImageDataset.extract_from_mask(
        image, np.ones(image.shape[:2], dtype=bool))

    if network.convex:  # if the network is convex, only evaluate on x and y features
        features = features[:, :2]

    network.eval()
    with torch.no_grad():
        predictions = torch.round(
            network(torch.tensor(features))).detach().numpy().reshape(image.shape[:2])

    return image, predictions
