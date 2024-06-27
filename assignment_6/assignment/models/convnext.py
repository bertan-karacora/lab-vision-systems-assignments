import torch


class LayerNorm2d(torch.nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNextHead(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        name_layer_norm="LayerNorm2d",
        use_bias=True,
    ):
        super().__init__()

        self.head = torch.nn.Sequential(
            LayerNorm2d(num_channels_in, eps=1e-6),
            torch.nn.Flatten(1),
            torch.nn.Linear(num_channels_in, num_channels_out, bias=use_bias),
        )

    def forward(self, x):
        y = self.head(x)
        return y
