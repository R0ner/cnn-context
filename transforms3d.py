import os
from random import choice, randint, uniform
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Tuple, Type, Union)

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from scipy.ndimage import rotate
from torch import nn
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._color import _max_value


class RandomAxisFlip(nn.Module):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, inpt: Any) -> Any:
        return randomaxisflip(inpt, self.axis)


class ToTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inpt: Any) -> Any:
        return totensor(inpt)


class IntensityJitter(nn.Module):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__()
        self.brightness = self.to_factors(brightness, center=0)
        self.contrast = self.to_factors(contrast, center=1)

    @staticmethod
    def to_factors(value, center=0):
        if isinstance(value, float):
            return (center - value, center + value)
        elif isinstance(value, tuple):
            return (center + value[0], center + value[1])
        else:
            raise TypeError

    @staticmethod
    def get_params(
        brightness: Optional[List[float]], contrast: Optional[List[float]]
    ) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(2)

        b = (
            None
            if brightness is None
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        )
        c = (
            None
            if contrast is None
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        )

        return fn_idx, b, c

    def forward(self, inpt: Any) -> Any:
        fn_idx, brightness_factor, contrast_factor = self.get_params(
            self.brightness, self.contrast
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                inpt = adjust_brightness(inpt, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                inpt = adjust_contrast(inpt, contrast_factor)

        return inpt


class RollJitter(nn.Module):
    def __init__(
        self, shifts: Union[int, tuple[int]], dims: Union[int, tuple[int]]
    ) -> None:
        super().__init__()
        self.dims = dims
        self.shifts = shifts

        if isinstance(dims, int):
            self.dims = (dims,)
        if isinstance(shifts, int):
            self.shifts = tuple(len(self.dims) * [shifts])

        assert len(self.dims) == len(
            self.shifts
        ), f"'dims' and 'shifts' should match, but got shifts='{self.shifts}' and dims='{self.dims}'"

    def forward(self, inpt: Any) -> Any:
        shifts = tuple([randint(-shift, shift) for shift in self.shifts])
        return torch.roll(inpt, shifts=shifts, dims=self.dims)


class RandomRotation(nn.Module):
    def __init__(self, angles: tuple[float], **kwargs) -> None:
        super().__init__()
        self.angles = angles
        self.kwargs = kwargs

    def forward(self, inpt: Any) -> Any:
        inpt = inpt.numpy()
        angles = tuple([uniform(-angle, angle) for angle in self.angles])
        for angle, axes in zip(angles, ((-3, -2), (-3, -1), (-2, -1))):
            if abs(angle) > 1e-4:
                inpt = rotate(inpt, angle, axes, reshape=False, **self.kwargs)
        return torch.from_numpy(inpt)


class Standardize(nn.Module):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, inpt: Any) -> Any:
        return (inpt - self.mean) / self.std


class StandardizeInv(Standardize):
    def __init__(self, mean, std) -> None:
        super().__init__(mean, std)

    def forward(self, inpt: Any) -> Any:
        return inpt * self.std + self.mean


def randomaxisflip(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if choice([True, False]):
        return torch.flip(volume, (axis - 3,))
    else:
        return volume


def totensor(inpt: Union[np.array, torch.Tensor]) -> torch.Tensor:
    if isinstance(inpt, np.ndarray):
        return torch.from_numpy(inpt)
    else:
        return inpt


def adjust_brightness(inpt: torch.Tensor, brightness_factor: float) -> torch.Tensor:

    fp = inpt.is_floating_point()
    bound = _max_value(inpt.dtype)
    output = inpt.add(brightness_factor * bound).clamp_(0, bound)
    return output if fp else output.to(inpt.dtype)


def adjust_contrast(inpt: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    fp = inpt.is_floating_point()
    bound = _max_value(inpt.dtype)
    output = inpt.mul(contrast_factor).clamp_(0, bound)
    return output if fp else output.to(inpt.dtype)
