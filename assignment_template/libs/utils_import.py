import inspect

import torch
import torchmetrics

# TODO: find a way to import all metrics (if not too costly) or something
import torchmetrics.classification
import torchmetrics.image.fid
import torchvision as tv
import torchvision.transforms.v2 as tv_transforms

import assignment.datasets as custom_datasets
import assignment.losses as custom_losses
import assignment.metrics as custom_metrics
import assignment.models as custom_models
import assignment.transforms as custom_transforms


def import_dataset(name):
    """Return dataset class if it exists in custom datasets or Torchvision."""
    modules = [custom_datasets, tv.datasets]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Dataset '{name}' not found")


def import_model(name):
    """Return model class or factory function if it exists in custom models or Torchvision."""
    modules = [custom_models, tv.models]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Model '{name}' not found")


def import_transform(name):
    """Return transform class if it exists in custom transform or Torchvision."""
    modules = [custom_transforms, tv_transforms]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Transform '{name}' not found")


def import_criterion(name):
    """Return loss class if it exists in custom losses or Pytorch."""
    modules = [custom_losses, torch.nn]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Loss '{name}' not found")


def import_metric(name):
    """Return metric class if it exists in custom metrics, custom losses, Pytorch, or Torchmetrics."""
    modules = [custom_metrics, custom_losses, torch.nn, torchmetrics, torchmetrics.classification, torchmetrics.image.fid]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Metric '{name}' not found")
