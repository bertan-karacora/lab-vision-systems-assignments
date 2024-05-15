from torch.utils.data import Dataset
import torchvision as tv

import assignment.libs.utils_data as utils_data


class SVHN(Dataset):
    def __init__(self, path, split="train", transform=None, transform_target=None, download=True, ratio_train2validate=0.83):
        self.dataset_tv = None
        self.download = download
        self.path = path
        self.ratio_train2validate = ratio_train2validate
        self.split = split
        self.transform = transform
        self.transform_target = transform_target

        self._initialize()

    def _initialize(self):
        self.dataset_tv = tv.datasets.SVHN(
            root=self.path,
            split="train" if self.split in ["train", "validate"] else "test",
            transform=self.transform,
            target_transform=self.transform_target,
            download=self.download,
        )

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
    Root location: {self.path}
    Split: {self.split}
    Transform: {self.transform}
    Transform of target: {self.transform_target}"""

        return s
