from pathlib import Path

import torch

import assignment.config as config
import assignment.libs.collations as collations
import assignment.libs.utils_import as utils_import
import assignment.libs.utils_model as utils_model


def create_transform(dict_transform):
    class_transform = utils_import.import_transform(dict_transform["name"])
    if dict_transform["name"] in ["Compose", "RandomApply", "RandomChoice", "RandomOrder"]:
        transform = class_transform(
            transforms=create_transforms(dict_transform["transforms"]),
            **dict_transform["kwargs"],
        )
    else:
        transform = class_transform(**dict_transform["kwargs"])

    return transform


def create_transforms(dicts_transforms):
    transforms = []
    for dict_transform in dicts_transforms:
        transform = create_transform(dict_transform)
        transforms += [transform]

    return transforms


def create_transform_dataset(split, target):
    dict_transform = config.DATA[split]["dataset"]["transform"][target]
    transform = create_transform(dict_transform) if dict_transform else None

    return transform


def create_dataset(split):
    name_dataset = config.DATA[split]["dataset"]["name"]

    class_dataset = utils_import.import_dataset(name_dataset)
    dataset_split = class_dataset(
        path=Path(config._PATH_DIR_DATA) / name_dataset.lower(),
        split=split,
        transform=create_transform_dataset(split, "features"),
        transform_target=create_transform_dataset(split, "target"),
        **config.DATA[split]["dataset"]["kwargs"],
    )

    return dataset_split


def create_collation(split):
    name_collation = config.DATA[split]["dataloader"]["collation"]["name"]
    create_collation_fn = getattr(collations, name_collation)
    collate_fn = create_collation_fn(**config.DATA[split]["dataloader"]["collation"]["kwargs"])

    return collate_fn


def create_dataloader(dataset_split, split):
    dataloader_split = torch.utils.data.DataLoader(
        dataset_split,
        num_workers=config._NUM_WORKERS_DATALOADING,
        collate_fn=create_collation(split) if "collation" in config.DATA[split]["dataloader"] else None,
        **config.DATA[split]["dataloader"]["kwargs"],
    )

    return dataloader_split


def create_dataset_and_dataloader(split):
    dataset_split = create_dataset(split)
    dataloader_split = create_dataloader(dataset_split, split)

    return dataset_split, dataloader_split


def create_model():
    class_model = utils_import.import_model(config.MODEL["name"])
    model = class_model(**config.MODEL["kwargs"]).eval()

    # TODO: This is bad.
    if "transfer" in config.MODEL:
        if "epochs_freeze" in config.MODEL["transfer"] and config.MODEL["transfer"]["epochs_freeze"] > 0:
            model = utils_model.freeze(model)

        for dict_layer in config.MODEL["transfer"]["layers"]:
            dict_model_layer = dict_layer["model"]
            class_model_layer = utils_import.import_model(dict_model_layer["name"])
            model_layer = class_model_layer(**dict_model_layer["kwargs"]).eval()

            setattr(model, dict_layer["name"], model_layer)

    return model


def create_criterion(dict_criterion=None):
    dict_criterion = dict_criterion or config.CRITERION

    class_criterion = utils_import.import_criterion(dict_criterion["name"])
    if dict_criterion["name"] in ["SumWeighted"]:
        criterion = class_criterion(
            modules=create_criteria(dict_criterion["modules"]),
            **dict_criterion["kwargs"],
        )
    else:
        criterion = class_criterion(**dict_criterion["kwargs"])

    return criterion


def create_criteria(dicts_criteria=None):
    criteria = []
    for dict_criterion in dicts_criteria:
        criterion = create_criterion(dict_criterion)
        criteria += [criterion]
    return criteria


def create_measurer(dict_measurer):
    class_measurer = utils_import.import_metric(dict_measurer["name"])
    measurer = class_measurer(**dict_measurer["kwargs"])
    return measurer


def create_measurers(split):
    measurers = []
    for dict_measurer in config.MEASURERS[split]:
        measurer = create_measurer(dict_measurer)
        measurers += [measurer]
    return measurers


def create_optimizer(params):
    class_optimizer = getattr(torch.optim, config.TRAINING["optimizer"]["name"])
    optimizer = class_optimizer(params, **config.TRAINING["optimizer"]["kwargs"])
    return optimizer


def create_scheduler(optimizer, dict_scheduler=None):
    dict_scheduler = dict_scheduler or config.TRAINING["scheduler"]

    class_scheduler = getattr(torch.optim.lr_scheduler, dict_scheduler["name"])
    if dict_scheduler["name"] in ["SequentialLR"]:
        scheduler = class_scheduler(
            optimizer=optimizer,
            schedulers=create_schedulers(optimizer, dict_scheduler["schedulers"]),
            **dict_scheduler["kwargs"],
        )
    else:
        scheduler = class_scheduler(optimizer, **dict_scheduler["kwargs"])

    return scheduler


def create_schedulers(optimizer, dicts_schedulers):
    schedulers = []
    for dict_scheduler in dicts_schedulers:
        scheduler = create_scheduler(optimizer, dict_scheduler)
        schedulers += [scheduler]
    return schedulers
