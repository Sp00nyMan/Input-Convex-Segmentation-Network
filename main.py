import os
from models import Network, RealNVP

from utils import utils
from utils.data_utils import create_dataloaders, ImageDataset
from models.real_nvp.utils import get_param_groups
from utils.training_utils import TrainingCenter

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

args = utils.create_parser().parse_args()


def prepare_dataset_for_convex(original_image):

    # Use pretrained plain network to acquire labels for each pixel
    plain = Network(5, 1)
    tc = TrainingCenter(plain, None,
                        model_name="plain", resume_mode="best")
    _, predictions = tc.inference(original_image)
    labels = torch.tensor(predictions.reshape((predictions.size, 1)))

    # Use Flow Model to transfer coordinates from XY-space to a Latent space
    flow = RealNVP(in_channels=2, mid_channels=32)
    tc = TrainingCenter(flow, None,
                        model_name="flow", resume_mode="best")

    dataset = ImageDataset(original_image, None, None,
                           train=False, ignore_rgb=True)

    with torch.no_grad():
        dataset.data = tc.model(torch.tensor(dataset.data))[0]
    dataset.labels = labels

    return dataset


if __name__ == "__main__":
    original = os.path.join(args.image_folder,
                            args.image_name + ".png")
    foreground = os.path.join(args.image_folder,
                              args.image_name + "_fore" + ".png")
    background = os.path.join(args.image_folder,
                              args.image_name + "_back" + ".png")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    device = torch.device(device)

    if args.model == "flow":
        model = RealNVP(2, 32, device)
    elif args.model == "convex":
        model = Network(2, 1, convex=True)
    else:
        model = Network(5, 1)

    if args.model == "flow":
        param_groups = get_param_groups(model,
                                        args.decay, norm_suffix="weight_g")
        optimizer = AdamW(param_groups, args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(),
                          args.learning_rate, weight_decay=args.decay)

    if args.model == "convex":
        data = prepare_dataset_for_convex(original)
    else:
        data = ImageDataset(original, foreground, background,
                            ignore_rgb=(args.model == "flow"),
                            train=(args.resume != "best"))

    scheduler = ReduceLROnPlateau(optimizer, "min", threshold=1e-2,
                                  verbose=True, patience=5)

    train_loader, test_loader = create_dataloaders(data,
                                                   args.batch_size, args.test_batch_size)

    tc = TrainingCenter(model, optimizer, scheduler, model_name=args.model,
                        resume_mode=args.resume, device=device)
    print(f"Starting the training: model='{args.model}'; "
          f"epochs={args.epochs}; device={device}")
    tc.train(args.epochs, train_loader, test_loader)
