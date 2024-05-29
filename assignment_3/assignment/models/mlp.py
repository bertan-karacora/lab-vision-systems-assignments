import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_channels_in=32 * 32 * 3,
        nums_channels_hidden=(2**6, 2**5),
        num_channels_out=10,
        name_layer_norm=None,
        name_layer_act="ReLU",
        inplace=None,
        use_bias=True,
        prob_dropout=None,
    ):
        super().__init__()
        kwargs_inplace = {"inplace": inplace} if inplace is not None else {}

        nums_channels = [num_channels_in] + list(nums_channels_hidden) + [num_channels_out]
        layers = [torch.nn.Flatten()]
        for num_channels_i, num_channels_o in zip(nums_channels[:-2], nums_channels[1:-1]):
            layers.append(torch.nn.Linear(num_channels_i, num_channels_o, bias=use_bias))
            if name_layer_norm is not None:
                layer_norm = getattr(torch.nn, name_layer_norm)
                layers.append(layer_norm(num_channels_o))
            if name_layer_act is not None:
                layer_act = getattr(torch.nn, name_layer_act)
                layers.append(layer_act(**kwargs_inplace))
            if prob_dropout is not None:
                layers.append(torch.nn.Dropout(prob_dropout, **kwargs_inplace))
        layers.append(torch.nn.Linear(nums_channels[-2], nums_channels[-1], bias=use_bias))

        self.head = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.head(x)
        return y
