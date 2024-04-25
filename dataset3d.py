import json
import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset

import transforms3d as T


class BNSet(Dataset):
    """Dataset class for the bugNIST dataset"""

    def __init__(self, data_dir, split, subset=None, transform=None):
        self.data_dir = data_dir
        self.split = split.lower()
        self.subset = subset
        self.transform = transform
        self.vol_dir = f"{self.data_dir}/train"

        with open(f'{data_dir}/single_bugs_split.json', 'r') as fp:
            self.datasplit = json.load(fp)

        self.volumes = []
        self.labels = []
        for idx, name in enumerate(sorted(os.listdir(self.vol_dir))):
            if self.subset is not None and name.lower() not in self.subset:
                continue
            bug_paths = list(
                filter(
                    lambda f: f.endswith(".tif"),
                    map(
                        lambda file: f"{self.vol_dir}/{name}/{file}",
                        sorted(os.listdir(f"{self.vol_dir}/{name}")),
                    ),
                )
            )
            self.volumes.extend(bug_paths)
            self.labels.extend(len(bug_paths) * [idx])
        
        self.volumes = [self.volumes[idx] for idx in self.datasplit[self.split]]
        self.labels = [self.labels[idx] for idx in self.datasplit[self.split]]

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
        
        self.mask_dir = f"{self.data_dir}/train_mask"

        self.masks = []
        
        for idx, name in enumerate(sorted(os.listdir(self.mask_dir))):
            if self.subset is not None and name.lower() not in self.subset:
                continue
            mask_paths = list(
                filter(
                    lambda f: f.endswith(".tif"),
                    map(
                        lambda file: f"{self.mask_dir}/{name}/{file}",
                        sorted(os.listdir(f"{self.mask_dir}/{name}")),
                    ),
                )
            )
            self.masks.extend(mask_paths)
        self.masks = [self.masks[idx] for idx in self.datasplit[self.split]]
        
    def __getitem__(self, item):
        volume = imread(self.volumes[item])[np.newaxis]
        mask = imread(self.masks[item])[np.newaxis].astype(np.uint8)

        if self.transform_shared is not None:
            volume, mask = self.transform_shared(volume, mask)
            mask = mask.bool()

        if self.transform_vol is not None:
            volume = self.transform_vol(volume)
        
        return volume, self.labels[item], mask


def get_dloader_mask(split, batch_size, data_dir="data/BugNIST_DATA", subset=None, **kwargs):
    split = split.lower()
    shuffle = False
    if split == "train":
        dset = BNSetMasks(
            data_dir,
            split,
            subset=subset,
            transform_shared=transform_shared_augment,
            transform_vol=transform_vol_augment,
        )
        shuffle = True
    else:
        dset = BNSetMasks(
            data_dir,
            split,
            subset=subset,
            transform_shared=transform_shared,
            transform_vol=None,
        )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )
    return dloader


transform_shared = transforms.Compose(
    [
        T.ToTensor()
    ]
)

transform_vol = transforms.Compose(
    [
        transforms.ToDtype(torch.float32, scale=True),
        T.Standardize(0.0142, 0.0670)
    ]
)

transform_shared_augment = transforms.Compose(
    [
        T.ToTensor(),
        T.RandomAxisFlip(0),
        T.RandomAxisFlip(1),
        T.RandomAxisFlip(2),
        transforms.RandomApply([T.RollJitter((6, 3, 3), (-3, -2, -1))], p=0.5),
        transforms.RandomApply([T.RandomRotation((6, 6, 360))], p=0.2),
        
    ]
)

transform_vol_augment = transforms.Compose(
    [
        transforms.RandomApply([T.IntensityJitter(0.1, 0.1)], p=0.3),
        transform_vol
    ]
)
