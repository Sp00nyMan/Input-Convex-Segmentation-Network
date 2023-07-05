import numpy as np
import argparse
import os
import shutil
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch


def create_parser():
    parser = argparse.ArgumentParser(description="Trains a simple FC-network on a toy dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="plain",
        choices=["plain", "convex", "flow"]
    )
    parser.add_argument(
        "--image-name",
        "-i",
        type=str,
        required=True
    )
    parser.add_argument(
        "--image-folder",
        "-d",
        type=str,
        default=r".data"
    )
    # TRAINING PARAMETERS
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="Number of training epochs.")
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Train batch size.")
    parser.add_argument(
        "--test-batch-size",
        "-tb",
        type=int,
        default=1,
        help="Test Batch size.")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="Initial learning rate")
    parser.add_argument(
        "--decay",
        "-wd",
        type=float,
        default=5e-4,
        help="Weight decay (L2 penalty).")
    # CHECKPOINT OPTIONS
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=r".snapshots",
        help="Folder to save checkpoints.")
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        choices=[None, "best", "checkpoint"],
        help="Checkpoint path for resume/test.")
    return parser


class LogWriter:
    LOG_HEADER = "epoch,time(s),train_loss,test_loss,test_acc(%)\n"

    def __init__(self, path: str, model_name: str = ""):
        if not path:
            raise ValueError(f'Folder "{path}" is invalid')

        self.log_folder = path
        self.log_file = os.path.join(
            self.log_folder, "log" + (f"_{model_name}" if model_name else "") + ".csv")

        with open(self.log_file, 'w') as f:
            f.write(LogWriter.LOG_HEADER)

        self.tb_writer = SummaryWriter(os.path.join(
            self.log_folder, "tensorboard", model_name))

    def log(self, epoch: int, start_time: float, train_loss: float, test_loss: float, test_acc: float):
        epoch += 1
        elapsed_time = time() - start_time

        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{elapsed_time},{train_loss},{test_loss},{test_acc}")

        print(f"Epoch {epoch} | Time {elapsed_time:.3g} | "
              f"Train Loss {train_loss:.4g} | Test Loss {test_loss:.3g} | "
              f"Test Acc {test_acc:.3g}")

        self.tb_writer.add_scalar("Train/Loss", train_loss, epoch)
        self.tb_writer.add_scalar("Test/Loss", test_loss, epoch)
        self.tb_writer.add_scalar("Test/Accuracy", test_acc, epoch)


class CheckpointManager:
    def __init__(self, path: str, model_name: str = ""):
        if not path:
            raise ValueError(f'Folder "{path}" is invalid')

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.isdir(path):
            raise FileExistsError(f'{path} is not a dir')

        self.save_folder = path
        self.best_loss = np.inf
        self.model_name = model_name

    def save(self, model, optimizer, epoch: int, test_loss: float):
        is_best = test_loss < self.best_loss
        self.best_loss = min(self.best_loss, test_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "test_loss": test_loss,
            "optimizer": optimizer.state_dict(),
        }

        save_path = os.path.join(
            self.save_folder, f"{self.model_name}_checkpoint.pth.tar")
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = os.path.join(
                self.save_folder, f"{self.model_name}_best.pth.tar")
            shutil.copyfile(save_path, best_path)

    def load(self, model, optimizer, best=True):
        """
            best: Load best model if True, otherwise - last checkpoint
        """

        postfix = "best" if best else "checkpoint"
        save_fle = os.path.join(self.save_folder,
                                f"{self.model_name}_{postfix}.pth.tar")

        if not os.path.isfile(save_fle):
            raise FileNotFoundError(f'Checkpoint "{save_fle}" does not exist')

        checkpoint = torch.load(save_fle)

        self.best_loss = checkpoint["test_loss"]
        model.load_state_dict(checkpoint["model"])
        if optimizer:  # In the inference no optimizer is required
            optimizer.load_state_dict(checkpoint["optimizer"])

        return model, optimizer, checkpoint["epoch"] + 1
