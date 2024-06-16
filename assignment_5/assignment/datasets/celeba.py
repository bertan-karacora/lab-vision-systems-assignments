from pathlib import Path

import torch
import torchvision as tv


class CelebA(torch.utils.data.Dataset):
    """Note: Bounding boxes are wrong."""

    split2splittv = {
        "all": "all",
        "training": "train",
        "validation": "valid",
        "test": "test",
    }

    def __init__(self, path, split, transform=None, transform_target=None, use_download=False, types_target=None):
        self.dataset_tv = None
        self.path = Path(path)
        self.split = split
        self.types_target = types_target or []
        self.transform = transform
        self.transform_target = transform_target
        self.use_download = use_download

        self._init()

    def _init(self):
        self.dataset_tv = tv.datasets.CelebA(
            root=self.path,
            split="all",
            transform=self.transform,
            target_transform=self.transform_target,
            download=self.use_download,
            target_type=self.types_target,
        )

    def __len__(self):
        length = len(self.dataset_tv)
        return length

    def __getitem__(self, index):
        features, target = self.dataset_tv[index]

        if not self.types_target:
            target = features

        return features, target

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of samples: {self.__len__()}
    Path: {self.path}
    Split: {self.split}
    Transform of samples: {self.transform}
    Transform of targets: {self.transform_target}"""
        return s
