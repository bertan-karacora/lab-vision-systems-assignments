import torch
import torchvision.transforms.v2 as tv_transforms


def unnormalize(features, mean, std):
    mean = torch.as_tensor(mean, dtype=features.dtype, device=features.device)
    std = torch.as_tensor(std, dtype=features.dtype, device=features.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return mean + features * std


class Unnormalize(tv_transforms.Transform):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std

    def _transform(self, features, params):
        features_transformed = unnormalize(features, mean=self.mean, std=self.std)
        return features_transformed
