from time import time

import torch
import numpy as np

import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models import Network, RealNVP
from models.real_nvp.real_nvp_loss import RealNVPLoss
from utils.utils import LogWriter, CheckpointManager
from utils.data_utils import ImageDataset
from tqdm import tqdm


class TrainingCenter:
    def __init__(self, model: Network | RealNVP | str, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
                 snapshots_folder: str = r".snapshots", model_name: str = "", resume_mode: str = None, device: str = "cpu"):
        self.checkpoint_manager = CheckpointManager(
            snapshots_folder, model_name)
        self.log_writer = LogWriter(snapshots_folder, model_name)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.start_epoch = 0

        self.device = device

        if resume_mode:
            self.model, self.optimizer, self.start_epoch = self.checkpoint_manager.load(
                self.model, self.optimizer, resume_mode == "best")
            print(f"Model restored from epoch {self.start_epoch}. "
                  f"Loss: {self.checkpoint_manager.best_loss:.3f}")

    def inference(self, path_to_image: str):
        image = ImageDataset.preprocess_image(path_to_image)
        features = ImageDataset.extract_from_mask(
            image, np.ones(image.shape[:2], dtype=bool))

        if self.model.convex:
            features = features[:, :2]

        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(features)).detach().numpy()
            predictions = np.reshape(output, image.shape[:2])

        return image, predictions

    def train(self, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        self.model.to(self.device)

        print(f"Starting training from epoch: {self.start_epoch}")
        for epoch in range(self.start_epoch, epochs):
            start_time = time()

            train_losses = self._train_epoch(train_loader)
            avg_loss = np.mean(train_losses)

            test_loss, test_acc = self._test_epoch(test_loader)

            if self.lr_scheduler:
                self.lr_scheduler.step(test_loss)

            self.checkpoint_manager.save(
                self.model, self.optimizer, epoch, test_loss)
            self.log_writer.log(epoch, start_time,
                                avg_loss, test_loss, test_acc)

        print(f"The training is finished successfully! "
              f"The model is saved to {self.checkpoint_manager.save_folder}")

    def _train_epoch(self, data_loader: DataLoader) -> float:
        self.model.train()
        losses = []

        with tqdm(total=len(data_loader.dataset)) as progress:
            for inputs, targets in data_loader:
                inputs.to(self.device)
                self.optimizer.zero_grad()

                loss = self._train_step(inputs, targets)

                losses.append(loss)
                progress.set_postfix(loss=np.mean(losses))
                progress.update(len(inputs))

        return losses

    def _train_step(self, inputs, targets):
        if isinstance(self.model, RealNVP):
            return self._train_flow(inputs)
        if self.model.convex:
            return self._train_convex(inputs, targets)
        return self._train_plain(inputs, targets)

    def _train_plain(self, inputs, targets) -> float:
        prediction = self.model(inputs)
        loss = F.binary_cross_entropy(prediction, targets)

        loss.backward()
        self.optimizer.step()

        return float(loss)

    def _train_convex(self, inputs, targets) -> float:
        loss = self._train_plain(inputs, targets)
        self.model.step()
        return loss

    def _train_flow(self, inputs, loss_fn=RealNVPLoss()) -> float:
        z, sldj = self.model(inputs)
        loss = loss_fn(z, sldj)

        loss.backward()
        self.optimizer.step()

        return float(loss)

    def _test_epoch(self, data_loader: DataLoader):
        self.model.eval()

        losses = []
        corrects = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs.to(self.device)
                loss, correct = self._test_step(inputs, targets)

                losses.append(loss)
                corrects.append(correct)

        return np.mean(losses), np.mean(correct)

    def _test_step(self, inputs, targets):
        output = self.model(inputs)
        if isinstance(self.model, RealNVP):
            loss = RealNVPLoss()(*output)
            correct = 1
        else:
            loss = F.binary_cross_entropy(output, targets)
            correct = torch.round(output).eq(targets).sum().item()

        return loss, correct
