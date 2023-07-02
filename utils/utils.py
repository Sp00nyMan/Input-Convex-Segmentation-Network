from utils.data_utils import ImageDataset
import numpy as np
from models.model import Network
import argparse
import os
import shutil
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch


def create_parser():
    parser = argparse.ArgumentParser(description="Trains a simple FC-network on a toy dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     "--image",
    #     "-i",
    #     required=True,
    #     type=str,
    #     help="Path to the original image file.")
    # parser.add_argument(
    #     "--image-masked-fore",
    #     "-imf",
    #     required=True,
    #     type=str,
    #     help="Path to the image file with foreground scribbles.")
    # parser.add_argument(
    #     "--image-masked-back",
    #     "-imb",
    #     required=True,
    #     type=str,
    #     help="Path to the image file with background scribbles.")
    # TRAINING PARAMETERS
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=15,
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
        default=0.01,
        help="Initial learning rate")
    parser.add_argument(
        "--decay",
        "-wd",
        type=float,
        default=0.0005,
        help="Weight decay (L2 penalty).")
    # CHECKPOINT OPTIONS
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=".snapshots",
        help="Folder to save checkpoints.")
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default="",
        help="Checkpoint path for resume/test.")

    parser.add_argument(
        "--print-freq",
        type=int,
        default=50,
        help="Training loss print frequency (batches).")
    # ACCELERATION
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of pre-fetching threads.")
    return parser


LOG_FILE = None
writer = None
best_loss = np.inf


def setup_log(log_path: str, model_name: str = ""):
    if not log_path:
        raise ValueError("Provide the path to logs folder")

    global LOG_FILE, writer

    LOG_FILE = os.path.join(
        log_path, "log" + (f"_{model_name}" if model_name else "") + ".csv")
    with open(LOG_FILE, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_acc(%)\n')

    writer = SummaryWriter(os.path.join(log_path, "tensorboard", model_name))


def log(epoch: int, start_time: float, train_loss: float, test_loss: float, test_acc: float):
    epoch += 1
    elapsed_time = time() - start_time

    if not os.path.exists(LOG_FILE) or not os.path.isfile(LOG_FILE):
        raise FileNotFoundError(f"Log file '{LOG_FILE}' doesn't exist"
                                "Invoke setup_log() before using log()!")

    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{elapsed_time},{train_loss},{test_loss},{test_acc}")

    print(f"Epoch {epoch} | Time {elapsed_time:.3g} | "
          f"Train Loss {train_loss:.4g} | Test Loss {test_loss:.3g} | Test Acc {test_acc:.3g}")

    if not writer:
        raise ValueError("Tensorboard summary writer is not initialized."
                         "Invoke setup_log() before using log()!")

    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Test/Loss", test_loss, epoch)
    writer.add_scalar("Test/Accuracy", test_acc, epoch)


SAVE_PATH = None


def setup_checkpoint(save_path: str):
    if not save_path:
        raise ValueError("Provide the path to checkpoint save folder")
    global SAVE_PATH
    SAVE_PATH = save_path

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    if not os.path.isdir(SAVE_PATH):
        raise FileExistsError(f'{SAVE_PATH} is not a dir')


def checkpoint(net, optimizer, epoch: int, test_loss: float, model_name: str = "model"):
    global best_loss
    is_best = test_loss < best_loss
    best_loss = min(best_loss, test_loss)

    checkpoint = {
        "epoch": epoch,
        "network": net.state_dict(),
        "test_loss": test_loss,
        "optimizer": optimizer.state_dict(),
    }

    save_path = os.path.join(SAVE_PATH, f"{model_name}_checkpoint.pth.tar")
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(SAVE_PATH, f"{model_name}_best.pth.tar")
        shutil.copyfile(save_path, best_path)
