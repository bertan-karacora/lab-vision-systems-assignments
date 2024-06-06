from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torchvision as tv

import assignment.libs.utils_data as utils_data


class StanfordCars(Dataset):
    def __init__(self, path, split="train", transform=None, transform_target=None, use_download=False, ratio_validate_to_train=0.8):
        self.dataset_tv = None
        self.path = Path(path)
        self.ratio_validate_to_train = ratio_validate_to_train
        self.split = split
        self.transform = transform
        self.transform_target = transform_target
        self.use_download = use_download

        self._init()

    def _init(self):
        self.dataset_tv = tv.datasets.StanfordCars(
            root=self.path,
            split=self.split if self.split not in ["train", "validate"] else "train",
            transform=self.transform,
            target_transform=self.transform_target,
            download=self.use_download,
        )
        self.labelset = np.asarray(self.dataset_tv.classes)

        if self.split in ["train", "validate"]:
            subset_train, subset_validate = utils_data.split_into_train_and_validate(self.dataset_tv)
            self.dataset_tv = subset_train if self.split == "train" else subset_validate

    def __len__(self):
        length = len(self.dataset_tv)
        return length

    def __getitem__(self, index):
        features, target = self.dataset_tv[index]
        return features, target

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of datapoints: {self.__len__()}
    Path: {self.path}
    Split: {self.split}
    Transform: {self.transform}
    Transform of target: {self.transform_target}"""

        return s
