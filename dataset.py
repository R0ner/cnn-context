import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import pad


class HWSet(Dataset):
    """Dataset class for the Husky vs. Wolf dataset."""

    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split.lower()
        self.transform = transform
        self.val_img_dir = f"{self.data_dir}/val_images_hw"
        self.test_img_dir = f"{self.data_dir}/test_images_hw"
        self.train_img_dirs = [
            f"{self.data_dir}/train_images_{idx}_hw" for idx in range(4)
        ]
        self.train_img_dirs = [
            img_dir for img_dir in self.train_img_dirs if os.path.exists(img_dir)
        ]

        if self.split == "val":
            self.imgs = [
                f"{self.val_img_dir}/{img_file}"
                for img_file in os.listdir(self.val_img_dir)
            ]
        elif self.split == "train":
            self.imgs = list()
            for img_dir in self.train_img_dirs:
                self.imgs.extend(
                    [f"{img_dir}/{img_file}" for img_file in os.listdir(img_dir)]
                )
        elif self.split == "test":
            self.imgs = [
                f"{self.test_img_dir}/{img_file}"
                for img_file in os.listdir(self.test_img_dir)
            ]

        self.imgs = list(filter(os.path.isfile, self.imgs))
        self.imgs = sorted(self.imgs)
        self.labels = []
        for f in self.imgs:
            for label, (idx, name, obj_id) in enumerate(
                [(3, "siberian husky", "n02110185"), (205, "grey wolf", "n02114367")]
            ):
                if obj_id in f:
                    self.labels.append(label)
        self.labels = np.array(self.labels)

        assert len(self.imgs) == self.labels.size

    def __len__(self):
        return self.labels.size

    def __getitem__(self, item):
        img = Image.open(self.imgs[item])

        if self.transform is not None:
            img = self.transform(img)

        return (img, self.labels[item])


class HWSetMasks(HWSet):
    """Dataset class for returning images with their segmentation masks."""

    def __init__(self, data_dir, split, transform_shared=None, transform_img=None):
        super().__init__(data_dir, split, transform)
        self.transform_shared = transform_shared
        self.transform_img = transform_img

        self.val_mask_dir = f"{self.val_img_dir}_masks"
        self.train_mask_dirs = [f"{img_dir}_masks" for img_dir in self.train_img_dirs]
        if self.split == "val":
            self.masks = [
                f"{self.val_mask_dir}/{mask_file}"
                for mask_file in os.listdir(self.val_mask_dir)
            ]
        elif self.split == "train":
            self.masks = list()
            for mask_dir in self.train_mask_dirs:
                self.masks.extend(
                    [f"{mask_dir}/{mask_file}" for mask_file in os.listdir(mask_dir)]
                )
        elif self.split == "test":
            self.masks = [
                f"{self.test_mask_dir}/{mask_file}"
                for mask_file in os.listdir(self.test_mask_dir)
            ]
        self.masks = list(filter(os.path.isfile, self.masks))
        self.masks = sorted(self.masks)

    def __getitem__(self, item):
        img = Image.open(self.imgs[item]).convert("RGB")
        mask = Image.open(self.masks[item])

        if self.transform_shared is not None:
            img, mask = self.transform_shared(img, mask)

        if self.transform_img is not None:
            img = self.transform_img(img)

        return img, self.labels[item], mask


def pad_collate_fn(data):
    """
    Args:
        data: A list of tuples with (image, label, mask) where:
            - 'image' is a tensor of arbitrary shape (e.g., [C, H, W])
            - 'label' is a scalar (integer)
            - 'mask' is a tensor of the same shape as the image
    Returns:
        batch: A dictionary containing the batched tensors:
            - 'images': A tensor of shape [batch_size, C, max_H, max_W]
            - 'labels': A tensor of shape [batch_size]
            - 'masks': A tensor of shape [batch_size, C, max_H, max_W]
    """
    images, labels, masks = zip(*data)

    # Determine the maximum height and width in the batch
    max_H = max(image.shape[1] for image in images)
    max_W = max(image.shape[2] for image in images)

    # Initialize tensors for batched data
    batch_size = len(images)
    n_channels = images[0].shape[0]
    batch_images = []
    batch_masks = []

    # Pad and stack images and masks
    for i, (image, mask) in enumerate(zip(images, masks)):
        h, w = image.shape[1], image.shape[2]
        diff_h, diff_w = max_H - h, max_W - w
        pad_h, pad_w = diff_h // 2, diff_w // 2
        padding = [pad_w + int(diff_w % 2), pad_h + int(diff_h % 2), pad_w, pad_h]
        batch_images.append(pad(image, padding))
        batch_masks.append(pad(mask, padding))

    # Convert labels to a tensor
    batch_labels = torch.tensor(labels)

    return (torch.stack(batch_images), batch_labels, torch.stack(batch_masks))


def get_dloader(split, batch_size, data_dir="data", **kwargs):
    split = split.lower()
    shuffle = False
    if split == "train":
        dset = HWSetMasks(
            data_dir,
            split,
            transform_shared=transform_shared_augment,
            transform_img=transform_img_augment,
        )
        shuffle = True
    elif split == "val":
        dset = HWSetMasks(
            data_dir,
            split,
            transform_shared=transform_shared,
            transform_img=transform_img,
        )
    elif split == "test":
        dset = HWSetMasks(
            data_dir,
            split,
            transform_shared=transform_shared,
            transform_img=transform_img,
        )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
        **kwargs,
    )
    return dloader


transform = transforms.Compose(
    [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
normalize_inv = transforms.Compose(
    [
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
        ),
        transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1.0, 1.0, 1.0)),
    ]
)  # For visualization purposes.

totensor = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)

transform_shared = transforms.Compose(
    [transforms.Resize(256, antialias=True), totensor]
)

transform_img = transforms.Compose(
    [
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_shared_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(60, expand=False)], p=0.3),
        transforms.RandomApply(
            [
                transforms.RandomResizedCrop(
                    256, scale=(0.3, 1), ratio=(3 / 4, 4 / 3), antialias=True
                )
            ],
            p=0.3,
        ),
        transform_shared,
    ]
)

transform_img_augment = transforms.Compose(
    [
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, hue=0.3, saturation=0.3
                )
            ],
            p=0.4,
        ),
        transforms.RandomGrayscale(p=0.05),
        # transforms.RandomApply(
        #     [
        #         transforms.GaussianBlur(kernel_size=7, sigma=(.1, 1.5))
        #     ],
        #     p=0.05
        # ),
        transform_img,
    ]
)
