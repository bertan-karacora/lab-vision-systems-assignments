import torch
import torchvision.transforms.v2 as tv_transforms

import assignment.libs.utils_import as utils_import
from assignment.models.cnn import BlockConv2d, BlockConvTranspose2d


class GeneratorDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, nums_channels_hidden, use_normalize=False, kwargs_normalize=None):
        super().__init__()

        self.body = None
        self.kwargs_normalize = kwargs_normalize
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.use_normalize = use_normalize

        self._init()

    def _init(self):
        nums_channels_body = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConvTranspose2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=1 if i == 0 else 2,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 1 else None,
                        name_layer_act="ReLU" if i != len(nums_channels_body) - 1 else "Tanh",
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input):
        output = input.view(*input.shape, 1, 1)
        output = self.body(output)
        # Map to [0, 1]
        output = output * 0.5 + 0.5
        if self.use_normalize:
            output = tv_transforms.functional.normalize(output, **self.kwargs_normalize)
        return output


class DiscriminatorDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, nums_channels_hidden, prob_dropout=None):
        super().__init__()

        self.body = None
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        nums_channels_body = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=2 if i != len(nums_channels_body) - 1 else 4,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 1 else None,
                        name_layer_act="LeakyReLU" if i != len(nums_channels_body) - 1 else "Sigmoid",
                        kwargs_act=dict(negative_slope=0.2) if i != len(nums_channels_body) - 1 else None,
                        dropout=self.prob_dropout if i != len(nums_channels_body) - 1 else None,
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input):
        output = self.body(input)
        return output

    @torch.no_grad()
    def sample(self, num_samples=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        code_latent = torch.randn(num_samples, self.num_channels_latent, device=device)
        samples = self(code_latent)
        return samples


# class GAN(torch.nn.Module):
#     def __init__(
#         self,
#         num_channels_latent,
#         num_channels_real,
#         num_channels_out,
#         name_generator,
#         kwargs_generator,
#         name_discriminator,
#         kwargs_discriminator,
#     ):
#         super().__init__()

#         self.discriminator = None
#         self.generator = None
#         self.kwargs_discriminator = kwargs_discriminator or {}
#         self.kwargs_generator = kwargs_generator or {}
#         self.name_discriminator = name_discriminator
#         self.name_generator = name_generator
#         self.num_channels_latent = num_channels_latent
#         self.num_channels_real = num_channels_real
#         self.num_channels_out = num_channels_out

#         self._init()

#     def _init(self):
#         class_model = utils_import.import_model(self.name_generator)
#         self.generator = class_model(self.num_channels_latent, self.num_channels_real, **self.kwargs_generator)

#         class_model = utils_import.import_model(self.name_discriminator)
#         self.discriminator = class_model(self.num_channels_real, self.num_channels_out, **self.kwargs_discriminator)

#     def discriminate(self, input):
#         output = self.discriminator(input)
#         return output

#     def generate(self, input):
#         output = self.generator(input)
#         return output

#     def forward(self, input):
#         output = self.generate(input)
#         output = self.discriminate(output)
#         return output

#     @torch.no_grad()
#     def sample(self, num_samples=16):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         code_latent = torch.randn(num_samples, self.num_channels_latent, device=device)
#         samples = self.generate(code_latent)
#         return samples
