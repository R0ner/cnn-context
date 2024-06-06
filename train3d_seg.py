import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchmetrics
from monai import losses
from monai.networks import nets
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from dataset3d import get_dloader_noise
from util3d import get_obj_score3d, get_saliency3d

# Random seed
seed = 191510

data_dir = "/work3/s191510/data/BugNIST_DATA"
name_legend = {
    "ac": "brown_cricket",
    "bc": "black_cricket",
    "bf": "blow_fly",
    "bl": "buffalo_bettle_larva",
    "bp": "blow_fly_pupa",
    "cf": "curly-wing_fly",
    "gh": "grasshopper",
    "ma": "maggot",
    "ml": "mealworm",
    "pp": "green_bottle_fly_pupa",
    "sl": "soldier_fly_larva",
    "wo": "woodlice",
}

model_types = {
    "r18": "ResNet 18",
}


def get_model(model_type, device="cpu", seed=191510):
    torch.manual_seed(seed)
    model = nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(name_legend) + 1,  # Bug classes and background
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    model.to(device)
    return model


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("BN segmentation experiment hyperparameters", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)

    # Model parameters
    parser.add_argument(
        "--model_type", type=str, default="r18", help="Model type (r18 or r50)"
    )

    # Learning rate scheduling
    parser.add_argument("--lr_step", default=100, type=int)

    # Perlin
    parser.add_argument("--perlin", action="store_true")

    # Multiprocessing
    parser.add_argument("--num_workers", default=0, type=int)

    # Saving
    parser.add_argument("--save_every", default=25, type=int)

    # weights and biases
    parser.add_argument("--wandb", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BugNIST training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    print("Args:", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Set manual seed!
    torch.manual_seed(seed=seed)
    random.seed(seed)
    np.random.seed(seed)

    # Hyperparameters
    # Model
    model_type = args.model_type

    # Data
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Optimizer
    lr = args.lr
    weight_decay = args.wd

    # Training
    n_epochs = args.epochs

    save_every = args.save_every

    # Scheduler
    lr_step = args.lr_step

    # Perlin noise
    perlin = args.perlin

    # Use wandb
    use_wandb = args.wandb

    save_dir = (
        f"/work3/s191510/models/bn-seg-checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}"
    )

    while os.path.exists(save_dir):
        save_dir = (
            f"/work3/s191510/models/bn-seg-checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}"
        )
    save_dir_model = f"{save_dir}/cpts"

    for dir in (save_dir, save_dir_model):
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Save args.
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=6)

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BugNIST-segmentation",
            # track hyperparameters and run metadata
            config={"architecture": model_types[model_type], "batch_size": batch_size},
        )

    subset = list(name_legend.keys())

    persistent_workers = num_workers > 0
    trainloader = get_dloader_noise(
        "train",
        batch_size=batch_size,
        data_dir=data_dir,
        subset=subset,
        seg=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    valloader = get_dloader_noise(
        "val",
        batch_size=batch_size,
        data_dir=data_dir,
        subset=subset,
        seg=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    model = get_model(model_type=model_type, device=device, seed=seed)

    ce_weight = torch.ones(len(name_legend) + 1, device=device)
    ce_weight[0] = 0.1
    criterion = losses.DiceCELoss(
        ce_weight=ce_weight, to_onehot_y=True, softmax=True, include_background=True
    )

    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)

    # Metrics
    if perlin:
        get_input = lambda volumes, masks, noise: volumes * masks + ~masks * noise
    else:
        get_input = lambda volumes, masks, noise: volumes * masks

    best = float("inf")

    stats = {}
    for epoch in range(n_epochs):
        metrics_train = {
            "loss": [],
            "accuracy": [],
        }
        metrics_val = {
            "loss": [],
            "accuracy": [],
        }

        print(f"Epoch {epoch}")
        model.train()
        count = 0
        for volumes, labels, masks, noise in tqdm(trainloader):
            target = (labels.view(-1, 1, 1, 1, 1) + 1) * masks
            out = model(get_input(volumes, masks, noise).to(device))
            
            optimizer.zero_grad()
            loss = criterion(out, target.to(device))

            loss.backward()
            optimizer.step()


            metrics_train["loss"].append(loss.cpu().detach().item())
            metrics_train["accuracy"].append((out.cpu().detach().max(1)[1].flatten(1) == target.flatten(1)).float().mean().item())

            count += 1
            if count > 10:
                break

        model.eval()
        count = 0
        for volumes, labels, masks, noise in tqdm(valloader):
            target = (labels.view(-1, 1, 1, 1, 1) + 1) * masks
            
            with torch.no_grad():
                out = model(get_input(volumes, masks, noise).to(device))
                loss = criterion(out, target.to(device))

            metrics_val["loss"].append(loss.cpu().detach().item())
            metrics_val["accuracy"].append((out.cpu().detach().max(1)[1].flatten(1) == target.flatten(1)).float().mean().item())

            count += 1
            if count > 10:
                break

        scheduler.step()

        performance = {
            "train_loss": np.mean(metrics_train["loss"]),
            "train_accuracy": np.mean(metrics_train["accuracy"]),
            "val_loss": np.mean(metrics_val["loss"]),
            "val_accuracy": np.mean(metrics_val["accuracy"]),
        }
        print(performance)
        stats[epoch] = performance

        # WandB
        if use_wandb:
            wandb.log(performance)

        suffix = ""
        if performance["val_loss"] < best:
            suffix = "_best"

        # Delete previous best.
        if len(suffix):
            for f in os.listdir(save_dir_model):
                if suffix in f:
                    os.remove(f"{save_dir_model}/{f}")

        best = min(best, performance["val_loss"])

        save = (
            ((epoch + 1) == n_epochs)
            or bool(len(suffix))
            or ((epoch + 1) % save_every == 0)
        )

        if save:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                f"{save_dir_model}/{model_type}_e{epoch}{suffix}.cpt",
            )

        if (epoch + 1) % 10 == 0:
            stats_df = pd.DataFrame.from_dict(stats, orient="index")
            stats_df.to_csv(f"{save_dir}/stats.csv")
        
    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.to_csv(f"{save_dir}/stats.csv")
