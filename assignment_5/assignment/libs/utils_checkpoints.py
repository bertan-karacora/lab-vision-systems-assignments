import torch

import assignment.libs.factory as factory


def save(trainer, epoch, name=None):
    path_dir_checkpoints = trainer.path_dir_exp / "checkpoints"

    filename = f"{name}.pth" if name is not None else f"epoch_{epoch}.pth"
    path_checkpoint = path_dir_checkpoints / filename
    torch.save(
        {
            "epoch": epoch,
            "state_dict_model": trainer.model.state_dict(),
            "state_dict_optimizer": trainer.optimizer.state_dict(),
            "state_dict_scheduler": "" if trainer.scheduler is None else trainer.scheduler.state_dict(),
        },
        path_checkpoint,
    )


def load(path):
    checkpoint = torch.load(path)

    epoch = None
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]

    model = None
    if "state_dict_model" in checkpoint:
        model = factory.create_model()
        model.load_state_dict(checkpoint["state_dict_model"])

    optimizer = None
    if "state_dict_optimizer" in checkpoint and model is not None:
        optimizer = factory.create_optimizer(model.parameters())
        optimizer.load_state_dict(checkpoint["state_dict_optimizer"])

    scheduler = None
    if "state_dict_scheduler" in checkpoint and optimizer is not None:
        scheduler = factory.create_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint["state_dict_scheduler"])

    return epoch, model, optimizer, scheduler


def load_model(path):
    checkpoint = torch.load(path)

    model = factory.create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
