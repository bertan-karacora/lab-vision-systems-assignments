import torch
import torchvision.transforms.v2 as tv_transforms


class Unnormalize(tv_transforms.Transform):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def _transform(self, features, params):
        features_transformed = self.mean[:, None, None] + features * self.std[:, None, None]
        return features_transformed
