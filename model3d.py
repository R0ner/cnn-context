import torch
from torch import nn


class CNN3d(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[
                *self.conv_layer(1, 4, kernel_size=7, stride=2, padding=3),
                *self.conv_layer(4, 8, kernel_size=3),
                nn.MaxPool3d(kernel_size=2, stride=2),
                *self.conv_layer(8, 16, kernel_size=3),
                nn.MaxPool3d(kernel_size=2, stride=2),
                *self.conv_layer(16, 32, kernel_size=3),
                nn.MaxPool3d(kernel_size=2, stride=2),
                *self.conv_layer(32, 32, kernel_size=3)
            ]
        )
        self.fc = nn.Linear(4096, num_classes)

    def conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
