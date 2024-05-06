import argparse
import json
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
from loss import SuperpixelCriterion
from scheduler import EarlyStopper, EarlyStopperSmooth, ReduceLROnPlateauSmooth
from util import DummyModel, eval_step, get_performance, train_step

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

    # Custom loss function
    parser.add_argument("--sp_loss", action="store_true")
    parser.add_argument(
        "--sp_weight",
        default=1.0,
        type=float,
        help="Weight given to the superpixel loss.",
    )
    parser.add_argument(
        "--sp_lw", type=str, default="constant", help="Layer weighting scheme: ['constant', 'geometric']"
    )
    parser.add_argument("--sp_normalize", action="store_true")
    parser.add_argument("--sp_binary", action="store_true")
    parser.add_argument("--sp_binary_th", default=0.5, type=float)
    parser.add_argument("--sp_mode", default='l2', type=str, help="One of ['l1', 'l2'] (see L1 and L2 norm).")

    # Model parameters
    parser.add_argument(
        "--model_type", type=str, default="r18", help="Model type (r18 or r50)"
    )

    # Learning rate scheduler and early stopping
    parser.add_argument("--lr_patience", default=20, type=int)
    parser.add_argument("--patience", default=40, type=int)
    parser.add_argument("--smooth_mode", default="", type=str)
    parser.add_argument("--n_smooth", default=10, type=int)
    parser.add_argument("--warmup", default=400, type=int)

    # Multiprocessing
    parser.add_argument("--num_workers", default=0, type=int)

    # Saving
    parser.add_argument("--save_every", default=100, type=int)
    
    # weights and biases
    parser.add_argument("--wandb", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW training script", parents=[get_args_parser()])
    args = parser.parse_args()

    print("Args:", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Hyperparameters
    # Model
    model_type = args.model_type
    assert model_type in model_types, "Invalid model type: " + model_type

    # Data
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Optimizer
    lr = args.lr

    # Training
    n_epochs = args.epochs

    # Learning rate scheduler
    factor = 0.1
    lr_patience = args.lr_patience

    # Early stopping
    patience = args.patience

    # Early stopping and lr scheduler smoothing
    smooth_mode = args.smooth_mode
    n_smooth = args.n_smooth
    smooth = len(smooth_mode) > 0
    warmup = args.warmup

    # Checkpoints
    save_every = 100

    # Use wandb
    use_wandb = args.wandb

    # Use sp loss
    sp_loss = args.sp_loss

    names = ("a", "b", "c")

    save_dir = f"models/hw-checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}"

    save_dir_models = {k: f"{save_dir}/{k}" for k in names}

    for dir in (save_dir, *save_dir_models.values()):
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    # Save args.
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=6)

    # Set manual seed!
    torch.manual_seed(seed=seed)

    # Get models
    models = {k: get_model(model_type, device=device, seed=seed) for k in names}

    # Loss function
    if not sp_loss:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = SuperpixelCriterion(
            model_type,
            sp_loss_weight=args.sp_weight,
            layer_weights=args.sp_lw,
            normalize=args.sp_normalize,
            binary=args.sp_binary,
            binary_threshold=args.sp_binary_th,
            mode=args.sp_mode,
            device=device,
        )

    # Optimizers
    optimizers = {
        k: torch.optim.Adam(model.parameters(), lr=lr) for k, model in models.items()
    }

    # Learning rate schedulers
    if smooth:
        get_lr_scheduler = lambda optimizer: ReduceLROnPlateauSmooth(
            optimizer,
            mode="min",
            smooth_mode=smooth_mode,
            n_smooth=n_smooth,
            factor=factor,
            patience=lr_patience,
        )
    else:
        get_lr_scheduler = lambda optimizer: ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=lr_patience
        )
    lr_schedulers = {
        k: get_lr_scheduler(optimizer) for k, optimizer in optimizers.items()
    }

    # Early stopping
    if smooth:
        get_early_stopper = lambda: EarlyStopperSmooth(
            mode="min", patience=patience, smooth_mode=smooth_mode, n_smooth=n_smooth
        )
    else:
        get_early_stopper = lambda: EarlyStopper(mode="min", patience=patience)
    earlystoppers = {k: get_early_stopper() for k in names}

    persistent_workers = num_workers > 0
    trainloader = get_dloader(
        "train",
        batch_size=batch_size,
        noise=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    valloader = get_dloader(
        "val",
        batch_size=1,
        noise=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    # WandB
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="HW-context",
            # track hyperparameters and run metadata
            config={"architecture": model_types[model_type], "batch_size": batch_size},
        )

    all_stats = {}
    completed = {k: False for k in names}

    best = {name: float("inf") for name in names}

    metrics_train_dict = lambda: {
        "loss_total": [],
        "loss_ce": [],
        "loss_features": [],
        "preds": [],
        "scores": [],
        "labels": [],
    }

    metrics_val_dict = lambda: {**metrics_train_dict(), "obj_scores": []}

    # Training loop
    print(f"Start training with model type: {model_type}")
    for epoch in range(n_epochs):
        metrics_train = {k: metrics_train_dict() for k in names}
        metrics_val = {k: metrics_val_dict() for k in names}

        print(f"Epoch {epoch}")

        # Train
        for model in models.values():
            model.train()
        for imgs, labels, masks, noise in tqdm(trainloader):
            train_step(
                models["a"],
                normalize_hw(imgs),
                labels,
                optimizers["a"],
                criterion,
                masks=masks,
                device=device,
                metrics=metrics_train["a"],
                return_features=sp_loss,
            )
            train_step(
                models["b"],
                normalize_hw_mask(imgs) * masks,
                labels,
                optimizers["b"],
                criterion,
                masks=masks,
                device=device,
                metrics=metrics_train["b"],
                return_features=sp_loss,
            )
            train_step(
                models["c"],
                normalize_hw_mask(imgs * masks + noise * (~masks)),
                labels,
                optimizers["c"],
                criterion,
                masks=masks,
                device=device,
                metrics=metrics_train["c"],
                return_features=sp_loss,
            )

        # Val
        for model in models.values():
            model.eval()
        for imgs, labels, masks, noise in tqdm(valloader):
            eval_step(
                models["a"],
                normalize_hw(imgs),
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val["a"],
                return_features=sp_loss,
            )
            eval_step(
                models["b"],
                normalize_hw_mask(imgs) * masks,
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val["b"],
                return_features=sp_loss,
            )
            eval_step(
                models["c"],
                normalize_hw_mask(imgs * masks + noise * (~masks)),
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val["c"],
                return_features=sp_loss,
            )

        # Calculate performance metrics
        log_stats = dict()
        suffix = {name: "" for name in names}
        stop = {name: False for name in names}
        for k in names:
            if completed[k]:
                continue
            performance_train = get_performance(metrics_train[k])
            performance_val = get_performance(metrics_val[k])

            print(f"Train performance: {performance_train}")
            print(f"Val performance: {performance_val}")

            if epoch >= warmup:
                lr_schedulers[k].step(performance_val["mean_loss_total"])
                stop[k] = earlystoppers[k](performance_val["mean_loss_total"])

            log_stats = (
                log_stats
                | {f"train/{p_k}_{k}": p_v for p_k, p_v in performance_train.items()}
                | {f"val/{p_k}_{k}": p_v for p_k, p_v in performance_val.items()}
                | {f"param/lr_{k}": optimizers[k].param_groups[-1]["lr"]}
            )

            if (
                performance_val["mean_loss_total"] < best[k]
                and (epoch + 1) > save_every
            ):
                suffix[k] = "_best"
            best[k] = min(best[k], performance_val["mean_loss_total"])

        # Track stats
        all_stats[epoch] = log_stats

        # WandB
        if use_wandb:
            wandb.log(log_stats)

        # Delete previous best.
        for k in names:
            if len(suffix[k]):
                for f in os.listdir(save_dir_models[k]):
                    if suffix[k] in f:
                        os.remove(f"{save_dir_models[k]}/{f}")

        # Save checkpoints
        save = (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs

        for k in names:
            if (save or len(suffix[k]) or stop[k]) and not completed[k]:
                torch.save(
                    {"epoch": epoch, "model_state_dict": models[k].state_dict()},
                    f"{save_dir_models[k]}/{model_type}_e{epoch}{suffix[k]}.cpt",
                )

        # Save stats
        if save:
            stats_df = pd.DataFrame.from_dict(all_stats, orient="index")
            stats_df.to_csv(f"{save_dir}/stats.csv")

        # If stop, set completed flag.
        for k in names:
            if stop[k] and not completed[k]:
                completed[k] = True
                models[k] = DummyModel()

        # Break if all models converged.
        if all(completed.values()):
            break

    stats_df = pd.DataFrame.from_dict(all_stats, orient="index")
    stats_df.to_csv(f"{save_dir}/stats.csv")
