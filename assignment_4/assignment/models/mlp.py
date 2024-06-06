import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        nums_channels_hidden,
        num_channels_out,
        name_layer_norm=None,
        name_layer_act="ReLU",
        inplace=None,
        use_bias=True,
        prob_dropout=None,
    ):
        super().__init__()

        self.head = None
        self.inplace = inplace
        self.name_layer_act = name_layer_act
        self.name_layer_norm = name_layer_norm
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.nums_channels_hidden = nums_channels_hidden
        self.prob_dropout = prob_dropout
        self.use_bias = use_bias

        self._init()

    def _init(self):
        self._init_head()

    def _init_head(self):
        kwargs_inplace = {"inplace": self.inplace} if self.inplace is not None else {}

        nums_channels = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]
        layers = [torch.nn.Flatten()]
        for num_channels_i, num_channels_o in zip(nums_channels[:-2], nums_channels[1:-1]):
            layers += [torch.nn.Linear(num_channels_i, num_channels_o, bias=self.use_bias)]

            if self.name_layer_norm is not None:
                layer_norm = getattr(torch.nn, self.name_layer_norm)
                layers += [layer_norm(num_channels_o)]

            if self.name_layer_act is not None:
                layer_act = getattr(torch.nn, self.name_layer_act)
                layers += [layer_act(**kwargs_inplace)]

            if self.prob_dropout is not None:
                layers += [torch.nn.Dropout(self.prob_dropout, **kwargs_inplace)]
        layers += [torch.nn.Linear(nums_channels[-2], nums_channels[-1], bias=self.use_bias)]

        self.head = torch.nn.Sequential(*layers)

    def forward(self, input):
        output = self.head(input)
        return output