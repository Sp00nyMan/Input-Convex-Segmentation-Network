from models import Network, RealNVP

from utils import utils
from utils.data_utils import create_dataloaders, ImageDataset
from models.real_nvp.utils import get_param_groups
from utils.training_utils import TrainingCenter

import torch
from torch.optim import AdamW

args = utils.create_parser().parse_args()

args.learning_rate = 1e-3
# args.resume = r".snapshots/real_nvp_checkpoint.pth.tar"


if __name__ == "__main__":
    data = ImageDataset(r"data/bf.png", r"data/bf_fore.png",
                        r"data/bf_back.png", ignore_rgb=True)
    train_loader, test_loader = create_dataloaders(
        data, args.batch_size, args.test_batch_size)

    net = RealNVP(in_channels=2, mid_channels=32)
    param_groups = get_param_groups(net, args.decay, norm_suffix="weight_g")
    optimizer = AdamW(param_groups, args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", threshold=1e-2)

    tc = TrainingCenter(net, optimizer, scheduler, model_name="real_nvp",
                        resume_mode="best" if args.resume else None)
    tc.train(50, train_loader, test_loader)
