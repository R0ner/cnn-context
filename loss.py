from typing import Any

import torch
from torch import nn


class SuperpixelWeights:
    def __init__(self, model_type: str, device: str = "cpu") -> None:
        self.model_type = model_type
        self.device = device

        if self.model_type != "r18":
            raise NotImplementedError

        self.conv7x7 = nn.Conv2d(
            1, 1, kernel_size=7, stride=2, padding=3, bias=False
        ).to(device)
        self.conv7x7.weight = self.conv7x7.weight.requires_grad_(False)
        self.conv7x7.weight *= 0
        self.conv7x7.weight += 1 / (7**2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3x3 = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        ).to(device)
        self.conv3x3_s2 = nn.Conv2d(
            1, 1, kernel_size=3, stride=2, padding=1, bias=False
        ).to(device)

        for conv in (self.conv3x3, self.conv3x3_s2):
            conv.weight = conv.weight.requires_grad_(False)
            conv.weight *= 0
            conv.weight += 1 / (3**2)

        self.layer1 = 4 * [self.conv3x3]
        self.layer234 = [self.conv3x3_s2, self.conv3x3] + 2 * [self.conv3x3]

        self.layer1 = nn.Sequential(*self.layer1)
        self.layer234 = nn.Sequential(*self.layer234)

    def __call__(self, masks: torch.tensor) -> list[torch.tensor]:
        sp_weights = []
        x = self.conv7x7(masks.float())
        sp_weights.append(1 - x)

        x = self.maxpool(x)
        x = self.layer1(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)

        return sp_weights


class SuperpixelCriterion:
    def __init__(self, model_type: str, device: str = "cpu") -> None:
        self.model_type = model_type
        self.device = device

        self.get_sp_weights = SuperpixelWeights(model_type, device)
        self.ce_criterion = nn.CrossEntropyLoss()  # Cross entropy

    def __call__(
        self, outs: list[torch.tensor], targets: torch.tensor, masks: torch.tensor
    ) -> torch.tensor:
        sp_weights = self.get_sp_weights(masks)

        loss = 0
        for sp, sp_w in zip(outs[:-1], sp_weights):
            loss += (sp_w * torch.square(sp)).sum() / (sp_w.sum() * sp_w.numel())

        loss /= len(sp_weights)
        loss_ce = self.ce_criterion(outs[-1], targets)
        loss += loss_ce
        return loss, loss_ce
