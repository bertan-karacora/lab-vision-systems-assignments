import torch

import assignment.config as config
import assignment.libs.factory as factory


def save(trainer, epoch, name=None):
    path_dir_checkpoints = trainer.path_dir_exp / "checkpoints"

    filename = f"{name}.pth" if name is not None else f"epoch_{epoch}.pth"
    path_checkpoint = path_dir_checkpoints / filename
    torch.save(
        {
            "epoch": epoch,
            "state_dict_model_discriminator": trainer.model_discriminator.state_dict(),
            "state_dict_model_generator": trainer.model_generator.state_dict(),
        },
        path_checkpoint,
    )


def load_model_discriminator(path):
    checkpoint = torch.load(path)

    model = factory.create_model(config.MODEL_DISCRIMINATOR)
    model.load_state_dict(checkpoint["state_dict_model_discriminator"])
    return model


def load_model_generator(path):
    checkpoint = torch.load(path)

    model = factory.create_model(config.MODEL_GENERATOR)
    model.load_state_dict(checkpoint["state_dict_model_generator"])
    return model
