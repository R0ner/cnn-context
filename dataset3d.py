import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset


class BNSet(Dataset):
    """Dataset class for the bugNIST dataset"""

    def __init__(self, data_dir, split, subset=None, transform=None):
        self.data_dir = data_dir
        self.split = split.lower()
        self.subset = subset
        self.transform = transform
        self.val_dir = f"{self.data_dir}/validation"
        self.test_dir = f"{self.data_dir}/test"
        self.train_dir = f"{self.data_dir}/train"

        self.volumes = []
        self.labels = []
        if self.split == "train":
            for idx, name in enumerate(sorted(os.listdir(self.train_dir))):
                if self.subset is not None and name.lower() not in self.subset:
                    continue
                bug_paths = list(
                    filter(
                        lambda f: f.endswith(".tif"),
                        map(
                            lambda file: f"{self.train_dir}/{name}/{file}",
                            sorted(os.listdir(f"{self.train_dir}/{name}")),
                        ),
                    )
                )
                self.volumes.extend(bug_paths)
                self.labels.extend(len(bug_paths) * [idx])
        elif self.split == "val":
            raise NotImplementedError
        elif self.split == "test":
            raise NotImplementedError

        self.labels = np.array(self.labels)

    def __len__(self):
        return self.labels.size

    def __getitem__(self, item):
        volume = imread(self.volumes[item])[np.newaxis]

        if self.transform is not None:
            volume = self.transform(volume)

        return volume, self.labels[item]


class BNSetMasks(BNSet):
    def __init__(self, data_dir, split, subset=None, transform_shared=None, transform_vol=None):
        super().__init__(data_dir, split, subset, None)
        self.transform_shared = transform_shared
        self.transform_vol = transform_vol
        
        self.val_mask_dir = f"{self.data_dir}/validation_mask"
        self.test_mask_dir = f"{self.data_dir}/test_mask"
        self.train_mask_dir = f"{self.data_dir}/train_mask"

        self.masks = []
        if self.split == "train":
            for idx, name in enumerate(sorted(os.listdir(self.train_mask_dir))):
                if self.subset is not None and name.lower() not in self.subset:
                    continue
                mask_paths = list(
                    filter(
                        lambda f: f.endswith(".tif"),
                        map(
                            lambda file: f"{self.train_mask_dir}/{name}/{file}",
                            sorted(os.listdir(f"{self.train_mask_dir}/{name}")),
                        ),
                    )
                )
                self.masks.extend(mask_paths)
        elif self.split == "val":
            raise NotImplementedError
        elif self.split == "test":
            raise NotImplementedError
        
    def __getitem__(self, item):
        volume = imread(self.volumes[item])[np.newaxis]
        mask = imread(self.masks[item])[np.newaxis].astype(np.uint8)

        if self.transform_shared is not None:
            volume, mask = self.transform_shared(volume, mask)
            mask = mask.bool()

        if self.transform_vol is not None:
            volume = self.transform_vol(volume)
        
        return volume, self.labels[item], mask