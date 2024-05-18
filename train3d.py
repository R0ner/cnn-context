import argparse
import os

import numpy as np
import torch
from monai.networks.nets import resnet10, resnet18
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

import wandb
from dataset3d import BNSet, BNSetMasks, get_dloader_mask, get_dloader_noise
from model3d import CNN3d
from util3d import get_obj_score3d, get_saliency3d, show_volume

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
    if model_type == "r18":
        model = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            no_max_pool=False,
            conv1_t_stride=2,
            num_classes=len(name_legend),
        )

    model.to(device)
    return model


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("HW experiment hyperparameters", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    # Model parameters
    parser.add_argument(
        "--model_type", type=str, default="r18", help="Model type (r18 or r50)"
    )

    # Perlin
    parser.add_argument("--perlin", action="store_true")

    # Multiprocessing
    parser.add_argument("--num_workers", default=0, type=int)

    # Saving
    parser.add_argument("--save_every", default=100, type=int)

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

    # Perlin noise
    perlin = args.perlin

    # Use wandb
    use_wandb = args.wandb

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BugNIST-classification",
            # track hyperparameters and run metadata
            config={"architecture": model_types[model_type], "batch_size": batch_size},
        )

    # subset = ["bc", "wo"]
    # subset = ["ac", "bc", "ml"]
    subset = list(name_legend.keys())

    persistent_workers = num_workers > 0
    trainloader = get_dloader_noise(
        "train",
        batch_size,
        data_dir=data_dir,
        subset=subset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    valloader = get_dloader_noise(
        "val",
        batch_size=1,
        data_dir=data_dir,
        subset=subset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    model = get_model(model_type=model_type, device=device, seed=seed)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=lr)

    if perlin:
        get_input = lambda volumes, masks, noise: volumes * masks + ~masks * noise
    else:
        get_input = lambda volumes, masks, noise: volumes * masks

    stats = {}
    for epoch in range(n_epochs):
        metrics_train = {
            "loss": [],
            "preds": [],
            "labels": [],
        }
        metrics_val = {
            "loss": [],
            "preds": [],
            "labels": [],
            "object_scores": [],
        }

        print(f"Epoch {epoch}")
        model.train()
        for volumes, labels, masks, noise in tqdm(trainloader):
            out = model(get_input(volumes, masks, noise).to(device))

            optimizer.zero_grad()
            loss = criterion(out, labels.type(torch.LongTensor).to(device))

            loss.backward()
            optimizer.step()

            _, indices = torch.max(out.cpu(), 1)

            metrics_train["loss"].append(loss.cpu().detach().item())
            metrics_train["preds"].append(indices.detach().numpy())
            metrics_train["labels"].append(labels.numpy())

        model.eval()
        for volumes, labels, masks, noise in tqdm(valloader):
            slc, score, indices, out = get_saliency3d(
                model, get_input(volumes, masks, noise), device=device
            )
            obj_score = get_obj_score3d(slc, masks)

            with torch.no_grad():
                loss = criterion(
                    out.to(device), labels.type(torch.LongTensor).to(device)
                )

            _, indices = torch.max(out.cpu(), 1)

            metrics_val["loss"].append(loss.cpu().detach().item())
            metrics_val["preds"].append(indices.detach().numpy())
            metrics_val["labels"].append(labels.numpy())
            metrics_val["object_scores"].append(obj_score)

        performance = {
            "train_loss": np.mean(metrics_train["loss"]),
            "train_accuracy": np.mean(
                np.concatenate(metrics_train["preds"])
                == np.concatenate(metrics_train["labels"])
            ).item(),
            "val_loss": np.mean(metrics_val["loss"]),
            "val_accuracy": np.mean(
                np.concatenate(metrics_val["preds"])
                == np.concatenate(metrics_val["labels"])
            ).item(),
            "obj_score": np.mean(metrics_val["object_scores"]),
        }
        print(performance)
        stats[epoch] = performance

        # WandB
        if use_wandb:
            wandb.log(performance)

        save = (epoch + 1) == n_epochs
        # if (save or len(suffix[k]) or stop[k]) and not completed[k]:
        #         torch.save(
        #             {"epoch": epoch, "model_state_dict": models[k].state_dict()},
        #             f"{save_dir_models[k]}/{model_type}_e{epoch}{suffix[k]}.cpt",
        #         )
