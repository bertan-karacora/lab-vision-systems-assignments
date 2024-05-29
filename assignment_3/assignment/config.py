import importlib.resources
from pathlib import Path
import pprint
import random

import numpy as np
import torch

import assignment.libs.utils_yaml as utils_yaml

_PATH_CONFIG_PACKAGE = str(importlib.resources.files(__package__) / "config.yaml")
_SEED_RNG = 42


def init():
    path_config = Path(_PATH_CONFIG_PACKAGE)
    attributes = utils_yaml.read(path_config)
    set_attributes(attributes)
    seed_package()

    print(f"Config loaded from {path_config}")


def get_attributes(private=False):
    config = {}
    for key, value in globals().items():
        if key.isupper() and (private or key[0] != "_"):
            config[key.lower()] = value

    return config


def set_attributes(config):
    # Use module globals for config attributes.
    # Globals are generally to be avoided but actually considered best practice for a Python package config.
    # https://stackoverflow.com/questions/5055042/whats-the-best-practice-using-a-settings-file-in-python
    # https://stackoverflow.com/questions/30556857/creating-a-static-class-with-no-instances
    for key, value in config.items():
        globals()[key.upper()] = value


def save(path, quiet=False):
    attributes = get_attributes()
    path_config = path / "config.yaml"
    utils_yaml.save(attributes, path_config)

    if not quiet:
        print(f"Config saved to {path_config}")


def dump():
    attributes = get_attributes(private=True)
    pprint.pprint(attributes)


def list_available():
    path_configs = importlib.resources.files(__package__) / "configs"
    files = sorted(path_configs.glob("**/*.yaml"))
    configs_available = [str(f.parent.relative_to(path_configs) / f.stem) for f in files]
    return configs_available


def set_config_preset(name, quiet=False):
    path_config = importlib.resources.files(__package__) / "configs" / f"{name}.yaml"
    attributes = utils_yaml.read(path_config)
    set_attributes(attributes)

    if not quiet:
        print(f"Config loaded from {path_config}")


def set_config_exp(path_dir_exp, quiet=False):
    path_config = path_dir_exp / "config.yaml"
    attributes = utils_yaml.read(path_config)
    set_attributes(attributes)

    if not quiet:
        print(f"Config loaded from {path_config}")


def seed_package():
    random.seed(_SEED_RNG)
    np.random.seed(_SEED_RNG)
    torch.manual_seed(_SEED_RNG)


# Run when module is imported (only once)
init()
