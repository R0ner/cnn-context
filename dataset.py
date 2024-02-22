import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HWSet(Dataset):
    """Dataset class for the Husky vs. Wolf dataset."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.val_img_dir = f"{self.data_dir}/val_images_hw"

        self.imgs = os.listdir(self.val_img_dir)

        self.labels = []
        for f in self.imgs:
            for label, (idx, name, obj_id) in enumerate([(3, "siberian husky", "n02110185"), (205, "grey wolf", "n02114367")]):
                if obj_id in f:
                    self.labels.append(label)
        self.labels = np.array(self.labels)
        
        assert len(self.imgs) == self.labels.size

    def __len__(self):
        return self.labels.size

    def __getitem__(self, item):
        img = Image.open(f"{self.val_img_dir}/{self.imgs[item]}")

        if self.transform is not None:
            img = self.transform(img)

        return (img, self.labels[item])


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
