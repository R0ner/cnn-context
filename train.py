import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import wandb
from torchvision.models import resnet18, resnet50

from dataset import get_dloader

# Random seed
seed = 191510

data_dir = "/data"
class_legend = ("Siberian Husky", "Grey Wolf")


def get_model(model_type, device="cpu", seed=191510):
    torch.manual_seed(seed)
    if model_type == 'r18':
        model = resnet18(weights=None)
    elif model_type == 'r50':
        model = resnet50(weights=None)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features, out_features=len(class_legend), bias=True
    )
    model.to(device)
    return model

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Hyperparameters
    # Model
    model_type = 'r18'
    
    # Data
    batch_size = 4

    # Optimizer
    lr = 1e-3

    model_a, model_b = get_model(model_type, device=device, seed=seed), get_model(model_type, device=device, seed=seed)

    trainloader = get_dloader('train', batch_size)