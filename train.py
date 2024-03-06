import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import wandb
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

from dataset import get_dloader
from util import eval_step, get_performance, train_step

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Hyperparameters
    # Model
    model_type = "r18"
    assert model_type in model_types, "Invalid model type: " + model_type

    # Data
    batch_size = 4
    num_workers = 0

    # Optimizer
    lr = 1e-3

    # Training
    n_epochs = 10

    model_a, model_b = get_model(model_type, device=device, seed=seed), get_model(
        model_type, device=device, seed=seed
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr)

    trainloader = get_dloader("train", batch_size, num_workers=num_workers)
    valloader = get_dloader("val", batch_size=1, num_workers=num_workers)

    # WandB
    wandb.init(
        # set the wandb project where this run will be logged
        project="HW-context",
        # track hyperparameters and run metadata
        config={
            "architecture": model_types[model_type]
        },
    )

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
        performance_train_a, performance_train_b = get_performance(
            metrics_train_a
        ), get_performance(metrics_train_b)
        performance_val_a, performance_val_b = get_performance(
            metrics_val_a
        ), get_performance(metrics_val_b)

        print(performance_train_a)
        print(performance_train_b)
        print(performance_val_a)
        print(performance_val_b)
        
        # WandB
        log_stats = {f'train/{k}_a': v for k, v in performance_train_a.items()}
        log_stats = log_stats | {f'train/{k}_b': v for k, v in performance_train_b.items()}
        log_stats = log_stats | {f'val/{k}_a': v for k, v in performance_val_a.items()}
        log_stats = log_stats | {f'val/{k}_b': v for k, v in performance_val_b.items()}
        wandb.log(log_stats)
