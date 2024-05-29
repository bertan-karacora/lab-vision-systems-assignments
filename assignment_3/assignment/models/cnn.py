import torch

from assignment.models.mlp import MLP


class BlockCNN2d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in=3,
        num_channels_out=3,
        shape_kernel_conv=(3, 3),
        kwargs_conv=None,
        name_layer_norm=None,
        name_layer_act="ReLU",
        name_layer_pool="MaxPool2d",
        shape_kernel_pool=(2, 2),
        kwargs_pool=None,
        inplace=None,
    ):
        kwargs_conv = kwargs_conv or {}
        kwargs_pool = kwargs_pool or {}
        kwargs_inplace = {"inplace": inplace} if inplace is not None else {}

        layers = [torch.nn.Conv2d(num_channels_in, num_channels_out, shape_kernel_conv, **kwargs_conv)]
        if name_layer_norm:
            layer_norm = getattr(torch.nn, name_layer_norm)
            layers.append(layer_norm(num_channels_out))
        if name_layer_act:
            layer_act = getattr(torch.nn, name_layer_act)
            layers.append(layer_act(**kwargs_inplace))
        if name_layer_pool:
            layer_pool = getattr(torch.nn, name_layer_pool)
            layers.append(layer_pool(shape_kernel_pool, **kwargs_pool))

        super().__init__(*layers)


class CNN2d(torch.nn.Module):
    def __init__(
        self,
        shape_input=(3, 32, 32),
        nums_channels_hidden_body=(16, 32),
        nums_channels_hidden_head=(2**6, 2**5),
        num_channels_out=10,
        kwargs_body=None,
        kwargs_head=None,
    ):
        super().__init__()
        kwargs_body = kwargs_body or {}
        kwargs_head = kwargs_head or {}

        nums_channels_body = [shape_input[0]] + list(nums_channels_hidden_body)
        blocks_body = []
        for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
            blocks_body.append(
                BlockCNN2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **kwargs_body,
                )
            )
        self.body = torch.nn.Sequential(*blocks_body)

        with torch.no_grad():
            dummy_input = torch.unsqueeze(torch.zeros(shape_input), 0)
            dummy_output_body = self.body(dummy_input)
            num_channels_in_head = dummy_output_body.nelement()

        self.head = MLP(
            num_channels_in=num_channels_in_head,
            nums_channels_hidden=nums_channels_hidden_head,
            num_channels_out=num_channels_out,
            **kwargs_head,
        )

    def forward(self, x):
        features_body = self.body(x)
        y = self.head(features_body)
        return y
