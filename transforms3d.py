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


class MultiTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def transform_(self, volume: Any, *args) -> Any:
        return volume

    def get_args_(self) -> tuple:
        return tuple()

    def transform(self, inpt: Any) -> Any:
        args = self.get_args_()
        if len(inpt) > 1:
            return tuple([self.transform_(vol, *args) for vol in inpt])
        else:
            return self.transform_(*inpt, *args)

    def forward(self, *inpt: Any) -> Any:
        return self.transform(inpt)


class ToTensor(MultiTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform_(self, volume: Any) -> Any:
        return totensor(volume)


class RandomAxisFlip(MultiTransform):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def transform_(self, volume: Any, flip: bool) -> Any:
        if flip:
            return axisflip(volume, self.axis)
        else:
            return volume

    def get_args_(self) -> tuple:
        return (choice([True, False]),)


class IntensityJitter(MultiTransform):
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

    def get_args_(self) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
        """
        Returns:
            fn_idx: The random order in which to apply brightness and contrast adjustments.
            b: The magnitude of the brightness adjustment.
            c: The magnitude of the contrast adjustment.
        """
        fn_idx = torch.randperm(2)

        b = (
            None
            if self.brightness is None
            else float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        )
        c = (
            None
            if self.contrast is None
            else float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        )

        return fn_idx, b, c
    
    def transform_(self, volume, fn_idx, brightness_factor, contrast_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                volume = adjust_brightness(volume, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                volume = adjust_contrast(volume, contrast_factor)
        return volume


class RollJitter(MultiTransform):
    def __init__(self, shifts: Union[int, tuple[int]], dims: Union[int, tuple[int]]) -> None:
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

    def get_args_(self) -> Tuple:
        return (tuple([randint(-shift, shift) for shift in self.shifts]), )
    
    def transform_(self, volume: Any, shifts: tuple[int]) -> Any:
        return torch.roll(volume, shifts=shifts, dims=self.dims)


class RandomRotation(MultiTransform):
    def __init__(self, angles: tuple[float], **kwargs) -> None:
        super().__init__()
        self.angles = angles
        self.kwargs = kwargs

    def get_args_(self) -> Tuple:
        return (tuple([uniform(-angle, angle) for angle in self.angles]), )
    
    def transform_(self, volume: Any, angles: tuple[float]):
        volume = volume.numpy()
        for angle, axes in zip(angles, ((-3, -2), (-3, -1), (-2, -1))):
            if abs(angle) > 1e-4:
                volume = rotate(volume, angle, axes, reshape=False, **self.kwargs)
        return torch.from_numpy(volume)


class Standardize(MultiTransform):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def transform_(self, volume: Any) -> Any:
        return (volume - self.mean) / self.std


class StandardizeInv(Standardize):
    def __init__(self, mean, std) -> None:
        super().__init__(mean, std)

    def transform_(self, volume: Any) -> Any:
        return volume * self.std + self.mean


def axisflip(volume: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.flip(volume, (axis - 3,))


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
