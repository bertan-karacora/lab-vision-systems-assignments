import torch

import assignment.config as config
import assignment.libs.utils_import as utils_import


def save(trainer, epoch, name=None):
    path_dir_checkpoints = trainer.path_dir_exp / "checkpoints"

    filename = f"{name}.pth" if name is not None else f"epoch_{epoch}.pth"
    path_checkpoint = path_dir_checkpoints / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": "" if trainer.scheduler is None else trainer.scheduler.state_dict(),
        },
        path_checkpoint,
    )


def load(path):
    class_model = utils_import.import_model(config.MODEL["name"])
    model = class_model(**config.MODEL["kwargs"]).eval()

    if "transfer" in config.MODEL:
        for dict_layer in config.MODEL["transfer"]["layers"]:
            dict_model_layer = dict_layer["model"]
            class_model_layer = utils_import.import_model(dict_model_layer["name"])
            model_layer = class_model_layer(**dict_model_layer["kwargs"]).eval()

            setattr(model, dict_layer["name"], model_layer)

    class_optimizer = getattr(torch.optim, config.TRAINING["optimizer"]["name"])
    optimizer = class_optimizer(model.parameters(), **config.TRAINING["optimizer"]["kwargs"])

    scheduler = None
    if "scheduler" in config.TRAINING:
        class_scheduler = getattr(torch.optim.lr_scheduler, config.TRAINING["scheduler"]["name"])
        scheduler = class_scheduler(optimizer, **config.TRAINING["scheduler"]["kwargs"])

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return epoch, model, optimizer, scheduler
