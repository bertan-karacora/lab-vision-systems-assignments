from functools import partial
import torch


class ConvNextHead(torch.nn.Module):
    def __init__(
        self,
        num_channels_in=32 * 32 * 3,
        num_channels_out=10,
        name_layer_norm="LayerNorm2d",
        use_bias=True,
    ):
        super().__init__()

        layer_norm = getattr(torch.nn, name_layer_norm)
        self.head = torch.nn.Sequential(
            partial(layer_norm, eps=1e-6),
            torch.nn.Flatten(1),
            torch.nn.Linear(num_channels_in, num_channels_out, bias=use_bias),
        )

    def forward(self, x):
        y = self.head(x)
        return y
