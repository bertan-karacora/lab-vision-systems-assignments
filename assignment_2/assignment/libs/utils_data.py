from pathlib import Path

import sklearn.model_selection as model_selection
import torch
import torchvision.transforms as tv_transforms

import assignment.config as config
import assignment.libs.utils_import as utils_import


def create_transforms(split, target):
    transforms = []
    for dict_transform in config.DATA[split]["dataset"]["transforms"][target]:
        class_transform = utils_import.import_transform(dict_transform["name"])
        transform = class_transform(**dict_transform["kwargs"])
        transforms += [transform]

    return transforms


def create_dataset(split):
    name_dataset = config.DATA[split]["dataset"]["name"]

    class_dataset = utils_import.import_dataset(name_dataset)
    dataset_split = class_dataset(
        path=Path(config._PATH_DIR_DATA) / name_dataset.lower(),
        split=split,
        transform=tv_transforms.Compose(create_transforms(split, "features")),
        transform_target=tv_transforms.Compose(create_transforms(split, "target")),
        **config.DATA[split]["dataset"]["kwargs"],
    )

    return dataset_split


def create_dataloader(dataset_split, split):
    dataloader_split = torch.utils.data.DataLoader(
        dataset_split,
        num_workers=config._NUM_WORKERS_DATALOADING,
        **config.DATA[split]["dataloader"]["kwargs"],
    )

    return dataloader_split


def create_dataset_and_dataloader(split):
    dataset_split = create_dataset(split)
    dataloader_split = create_dataloader(dataset_split, split)

    return dataset_split, dataloader_split


def split_into_train_and_validate(dataset, ratio_train2validate=0.8):
    indices = list(range(len(dataset)))
    indices_train, indices_validate = model_selection.train_test_split(indices, test_size=1.0 - ratio_train2validate)
    subset_train = torch.utils.data.Subset(dataset, indices_train)
    subset_validate = torch.utils.data.Subset(dataset, indices_validate)
    return subset_train, subset_validate


def sample(dataset, indices):
    """Sample indexed items from dataset by converting list of tuples of k elements into k separate lists."""
    lists = map(list, zip(*[dataset[i] for i in indices]))
    return lists
