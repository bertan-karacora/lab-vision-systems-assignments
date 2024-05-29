import torch
import torchvision.transforms.v2 as tv_transforms


def gaussian_noise(features, mean=0.0, std=1.0):
    features_noisy = features + torch.normal(mean, std, size=features.shape)

    return features_noisy


class GaussianNoise(tv_transforms.Transform):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()

        self.mean = mean
        self.std = std

    def _transform(self, features, params):
        features_transformed = gaussian_noise(features, mean=self.mean, std=self.std)
        return features_transformed
