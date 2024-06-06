import inspect

import torch
import torchvision as tv
import torchvision.transforms.v2 as tv_transforms

import assignment.datasets as custom_datasets
import assignment.models as custom_models
import assignment.transforms as custom_transforms


def import_dataset(name_dataset):
    """Return dataset class if it exists in custom datasets or in Torchvision."""
    modules = [custom_datasets, tv.datasets]

    for module in modules:
        if hasattr(module, name_dataset):
            class_found = getattr(module, name_dataset)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Dataset '{name_dataset}' not found")


def import_model(name_model):
    """Return model class if it exists in custom models or in Torchvision."""
    modules = [custom_models, tv.models]

    for module in modules:
        if hasattr(module, name_model):
            class_or_function_found = getattr(module, name_model)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Model '{name_model}' not found")


def import_transform(name):
    """Return transform class if it exists in custom transform or in Torchvision."""
    modules = [custom_transforms, tv_transforms]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Transform '{name}' not found")
