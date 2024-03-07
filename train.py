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
from dataset import get_dloader
from util import (DummyModel, EarlyStopper, eval_step, get_performance,
                  train_step)

# Random seed
seed = 191510

data_dir = "/data"
class_legend = ("Siberian Husky", "Grey Wolf")
model_types = {
        "r18": "ResNet 18",
        "r50": "Resnet 50"
}


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
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1000, type=int)

    # Model parameters
    parser.add_argument('--model_type', type=str, default="r18",
                        help="Model type (r18 or r50)")
    
    # Learning rate scheduler and early stopping
    parser.add_argument('--lr_patience', default=20, type=int)
    parser.add_argument('--patience', default=40, type=int)

    # weights and biases
    parser.add_argument('--wandb', action='store_true')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HW training script', parents=[get_args_parser()])
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
    for dir in (save_dir, save_dir_a, save_dir_b):
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    # Set manual seed!
    torch.manual_seed(seed=seed)

    # Get models
    model_a, model_b = get_model(model_type, device=device, seed=seed), get_model(
        model_type, device=device, seed=seed
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizers
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr)

    # Learning rate schedulers
    lr_scheduler_a = ReduceLROnPlateau(optimizer_a, mode='min', factor=factor, patience=lr_patience)
    lr_scheduler_b = ReduceLROnPlateau(optimizer_b, mode='min', factor=factor, patience=lr_patience)

    # Early stopping
    earlystopper_a = EarlyStopper(mode='min', patience=patience)
    earlystopper_b = EarlyStopper(mode='min', patience=patience)

    trainloader = get_dloader("train", batch_size, num_workers=num_workers)
    valloader = get_dloader("val", batch_size=1, num_workers=num_workers)

    # WandB
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="HW-context",
            # track hyperparameters and run metadata
            config={
                "architecture": model_types[model_type],
                "batch_size": batch_size
            },
        )
    
    all_stats = {}
    completed_a = False
    completed_b = False

    # Training loop
    print(f"Start training with model type: {model_type}")
    for epoch in range(n_epochs):
        metrics_train_a = {"loss": [], "preds": [], "scores": [], "labels": []}
        metrics_train_b = {"loss": [], "preds": [], "scores": [], "labels": []}
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

        print(f"Epoch {epoch}")

        # Train
        model_a.train()
        model_b.train()
        for imgs, labels, masks in tqdm(trainloader):
            train_step(
                model_a,
                imgs,
                labels,
                optimizer_a,
                criterion,
                device=device,
                metrics=metrics_train_a,
            )
            train_step(
                model_b,
                imgs * masks,
                labels,
                optimizer_b,
                criterion,
                device=device,
                metrics=metrics_train_b,
            )

        # Val
        model_a.eval()
        model_b.eval()
        for imgs, labels, masks in tqdm(valloader):
            eval_step(
                model_a,
                imgs,
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val_a,
            )
            eval_step(
                model_b,
                imgs * masks,
                labels,
                masks,
                criterion,
                device=device,
                metrics=metrics_val_b,
            )
        # Calculate performance metrics
        log_stats = dict()
        if not completed_a:
            performance_train_a = get_performance(metrics_train_a)
            performance_val_a = get_performance(metrics_val_a)
            
            print(performance_train_a)
            print(performance_val_a)
            
            lr_scheduler_a.step(performance_val_a['mean_loss'])
            stop_a = earlystopper_a(performance_val_a['mean_loss'])

            log_stats = log_stats | \
                {f'train/{k}_a': v for k, v in performance_train_a.items()} | \
                {f'val/{k}_a': v for k, v in performance_val_a.items()} | \
                {'param/lr_a': optimizer_a.param_groups[-1]['lr']}
        
        if not completed_b:
            performance_train_b = get_performance(metrics_train_b)
            performance_val_b = get_performance(metrics_val_b)
            
            print(performance_train_b)
            print(performance_val_b)
            
            lr_scheduler_b.step(performance_val_b['mean_loss'])
            stop_b = earlystopper_b(performance_val_b['mean_loss'])

            log_stats = log_stats | \
                {f'train/{k}_b': v for k, v in performance_train_b.items()} | \
                {f'val/{k}_b': v for k, v in performance_val_b.items()} | \
                {'param/lr_b': optimizer_b.param_groups[-1]['lr']}
            
        # Track stats
        all_stats[epoch] = log_stats

        # WandB
        if use_wandb:
            wandb.log(log_stats)
        
        # Save checkpoints
        if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
            if not completed_a:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_a.state_dict()
                }, f"{save_dir_a}/{model_type}_e{epoch}.cpt")
            if not completed_b:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_b.state_dict()
                }, f"{save_dir_b}/{model_type}_e{epoch}.cpt")
            stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
            stats_df.to_csv(f"{save_dir}/stats.csv")

        
        if stop_a and not completed_a:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_a.state_dict()
            }, f"{save_dir_a}/{model_type}_e{epoch}.cpt")
            completed_a = True
            model_a = DummyModel()
        
        if stop_b and not completed_b:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_b.state_dict()
            }, f"{save_dir_b}/{model_type}_e{epoch}.cpt")
            completed_b = True
            model_b = DummyModel()
        
        if completed_a and completed_b:
            break

    stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
    stats_df.to_csv(f"{save_dir}/stats.csv")
