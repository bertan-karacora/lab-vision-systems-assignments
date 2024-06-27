import torch

import assignment.config as config
import assignment.libs.utils_data as utils_data


@torch.no_grad()
def sample(model, split, num_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, target = utils_data.sample(split=split, num_samples=num_samples)

    model = model.to(device)
    features = features.to(device)
    target = target.to(device)

    output = model(features)

    return output, target


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model
