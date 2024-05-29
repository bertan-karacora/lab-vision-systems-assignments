from functools import partial
import torch


class ResNetHead(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        name_layer_norm="LayerNorm2d",
        use_bias=True,
    ):
        super().__init__()

        self.fc = torch.nn.Sequential(torch.nn.Linear(in_features=num_channels_in, out_features=num_channels_out, bias=use_bias))

    def forward(self, x):
        y = self.head(x)
        return y
