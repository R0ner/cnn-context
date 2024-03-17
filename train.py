import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

import wandb
from dataset import get_dloader, normalize_hw, normalize_hw_mask
from perlin import get_rgb_fractal_noise
from util import (DummyModel, EarlyStopper, eval_step, get_performance,
                  train_step)

# Random seed
seed = 191510

data_dir = "/data"
class_legend = ("Siberian Husky", "Grey Wolf")
model_types = {"r18": "ResNet 18", "r50": "Resnet 50"}


def get_model(model_type, device="cpu", seed=191510):
    torch.manual_seed(seed)
    if model_type == "r18":
        model = resnet18(weights=None)
    elif model_type == "r50":
        model = resnet50(weights=None)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features, out_features=len(class_legend), bias=True
    )
    model.to(device)
    return model


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("HW experiment hyperparameters", add_help=False)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    # Model parameters
    parser.add_argument(
        "--model_type", type=str, default="r18", help="Model type (r18 or r50)"
    )

    # Learning rate scheduler and early stopping
    parser.add_argument("--lr_patience", default=20, type=int)
    parser.add_argument("--patience", default=40, type=int)

    # weights and biases
    parser.add_argument("--wandb", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW training script", parents=[get_args_parser()])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Hyperparameters
    # Model
    model_type = args.model_type
    assert model_type in model_types, "Invalid model type: " + model_type

    # Data
    batch_size = args.batch_size
    num_workers = 0

    # Optimizer
    lr = args.lr

    # Training
    n_epochs = args.epochs

    # Learning rate scheduler
    factor = 0.1
    lr_patience = args.lr_patience

    # Early stopping
    patience = args.patience

    # Checkpoints
    save_every = 40

    # Use wandb
    use_wandb = args.wandb

    save_dir = f"models/hw-checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}"
    save_dir_a = f"{save_dir}/a"
    save_dir_b = f"{save_dir}/b"
    save_dir_c = f"{save_dir}/c"
    for dir in (save_dir, save_dir_a, save_dir_b, save_dir_c):
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Set manual seed!
    torch.manual_seed(seed=seed)

    # Get models
    model_a = get_model(model_type, device=device, seed=seed)
    model_b = get_model(model_type, device=device, seed=seed)
    model_c = get_model(model_type, device=device, seed=seed)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizers
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=lr)

    # Learning rate schedulers
    lr_scheduler_a = ReduceLROnPlateau(
        optimizer_a, mode="min", factor=factor, patience=lr_patience
    )
    lr_scheduler_b = ReduceLROnPlateau(
        optimizer_b, mode="min", factor=factor, patience=lr_patience
    )
    lr_scheduler_c = ReduceLROnPlateau(
        optimizer_c, mode="min", factor=factor, patience=lr_patience
    )

    # Early stopping
    earlystopper_a = EarlyStopper(mode="min", patience=patience)
    earlystopper_b = EarlyStopper(mode="min", patience=patience)
    earlystopper_c = EarlyStopper(mode="min", patience=patience)

    trainloader = get_dloader("train", batch_size=batch_size, noise=True, num_workers=num_workers)
    valloader = get_dloader("val", batch_size=1, noise=True, num_workers=num_workers)

    # WandB
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="HW-context",
            # track hyperparameters and run metadata
            config={"architecture": model_types[model_type], "batch_size": batch_size},
        )

    all_stats = {}
    completed_a = False
    completed_b = False
    completed_c = False

    names = ("a", "b", "c")

    best = {name: float("inf") for name in names}

    # Training loop
    print(f"Start training with model type: {model_type}")
    for epoch in range(n_epochs):
        metrics_train_a = {"loss": [], "preds": [], "scores": [], "labels": []}
        metrics_train_b = {"loss": [], "preds": [], "scores": [], "labels": []}
        metrics_train_c = {"loss": [], "preds": [], "scores": [], "labels": []}
        metrics_val_a = {
            "loss": [],
            "preds": [],
            "scores": [],
            "labels": [],
            "obj_scores": [],
        }
        metrics_val_b = {
            "loss": [],
            "preds": [],
            "scores": [],
            "labels": [],
            "obj_scores": [],
        }
        metrics_val_c = {
            "loss": [],
            "preds": [],
            "scores": [],
            "labels": [],
            "obj_scores": [],
        }

        print(f"Epoch {epoch}")

        # Train
        model_a.train()
        model_b.train()
        model_c.train()
        for imgs, labels, masks, noise in tqdm(trainloader):
            train_step(
                model_a,
                normalize_hw(imgs),
                labels,
                optimizer_a,
                criterion,
                device=device,
                metrics=metrics_train_a,
            )
            train_step(
                model_b,
                normalize_hw_mask(imgs) * masks,
                labels,
                optimizer_b,
                criterion,
                device=device,
                metrics=metrics_train_b,
            )
            train_step(
                model_c,
                normalize_hw_mask(imgs * masks + noise * (~masks)),
                labels,
                optimizer_c,
                criterion,
                device=device,
                metrics=metrics_train_c,
            )

        # Val
        model_a.eval()
        model_b.eval()
        model_c.eval()

        # # Ensure the same "random" noise every time.
        # gen = torch.Generator()
        # gen.manual_seed(seed)
        for imgs, labels, masks, noise in tqdm(valloader):
            eval_step(
                model_a,
                normalize_hw(imgs),
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val_a,
            )
            eval_step(
                model_b,
                normalize_hw_mask(imgs) * masks,
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val_b,
            )
            eval_step(
                model_c,
                normalize_hw_mask(imgs * masks + noise * (~masks)),
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val_c,
            )
        # Calculate performance metrics
        log_stats = dict()
        suffix = {name: "" for name in names}
        stop = {name: False for name in names}
        for (
            name,
            completed,
            metrics_train,
            metrics_val,
            lr_scheduler,
            earlystopper,
            optimizer,
        ) in zip(
            names,
            (completed_a, completed_b, completed_c),
            (metrics_train_a, metrics_train_b, metrics_train_c),
            (metrics_val_a, metrics_val_b, metrics_val_c),
            (lr_scheduler_a, lr_scheduler_b, lr_scheduler_c),
            (earlystopper_a, earlystopper_b, earlystopper_c),
            (optimizer_a, optimizer_b, optimizer_c),
        ):
            if not completed:
                performance_train = get_performance(metrics_train)
                performance_val = get_performance(metrics_val)

                print(performance_train)
                print(performance_val)

                lr_scheduler.step(performance_val["mean_loss"])
                stop[name] = earlystopper(performance_val["mean_loss"])

                log_stats = (
                    log_stats
                    | {f"train/{k}_{name}": v for k, v in performance_train.items()}
                    | {f"val/{k}_{name}": v for k, v in performance_val.items()}
                    | {f"param/lr_{name}": optimizer.param_groups[-1]["lr"]}
                )

                if (
                    performance_val["mean_loss"] < best[name]
                    and (epoch + 1) > save_every
                ):
                    suffix[name] = "_best"
            best[name] = min(best[name], performance_val["mean_loss"])

        # Track stats
        all_stats[epoch] = log_stats

        # WandB
        if use_wandb:
            wandb.log(log_stats)

        # Delete previous best.
        if len(suffix["a"]):
            for f in os.listdir(save_dir_a):
                if suffix["a"] in f:
                    os.remove(f"{save_dir_a}/{f}")
        if len(suffix["b"]):
            for f in os.listdir(save_dir_b):
                if suffix["b"] in f:
                    os.remove(f"{save_dir_b}/{f}")
        if len(suffix["c"]):
            for f in os.listdir(save_dir_c):
                if suffix["c"] in f:
                    os.remove(f"{save_dir_c}/{f}")

        # Save checkpoints
        save = (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs

        save_a = (save or len(suffix["a"]) or stop["a"]) and not completed_a
        save_b = (save or len(suffix["b"]) or stop["b"]) and not completed_b
        save_c = (save or len(suffix["c"]) or stop["c"]) and not completed_c

        if save_a:
            torch.save(
                {"epoch": epoch, "model_state_dict": model_a.state_dict()},
                f"{save_dir_a}/{model_type}_e{epoch}{suffix['a']}.cpt",
            )
        if save_b:
            torch.save(
                {"epoch": epoch, "model_state_dict": model_b.state_dict()},
                f"{save_dir_b}/{model_type}_e{epoch}{suffix['b']}.cpt",
            )
        if save_c:
            torch.save(
                {"epoch": epoch, "model_state_dict": model_c.state_dict()},
                f"{save_dir_c}/{model_type}_e{epoch}{suffix['c']}.cpt",
            )

        # Save stats
        if save:
            stats_df = pd.DataFrame.from_dict(all_stats, orient="index")
            stats_df.to_csv(f"{save_dir}/stats.csv")

        # If stop, set completed flag.
        if stop["a"] and not completed_a:
            completed_a = True
            model_a = DummyModel()

        if stop["b"] and not completed_b:
            completed_b = True
            model_b = DummyModel()

        if stop["c"] and not completed_c:
            completed_c = True
            model_c = DummyModel()

        # Break if all models converged.
        if completed_a and completed_b and completed_c:
            break

    stats_df = pd.DataFrame.from_dict(all_stats, orient="index")
    stats_df.to_csv(f"{save_dir}/stats.csv")
