import torch

from assignment.models.mlp import MLP


class BlockConv2d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv=(5, 5),
        kwargs_conv=None,
        name_layer_norm=None,
        name_layer_act=None,
        name_layer_pool=None,
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


class BlockDeconv2d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv=(5, 5),
        kwargs_conv=None,
        name_layer_norm=None,
        name_layer_act=None,
        name_layer_pool=None,
        shape_kernel_pool=(2, 2),
        kwargs_pool=None,
        inplace=None,
    ):
        kwargs_conv = kwargs_conv or {}
        kwargs_pool = kwargs_pool or {}
        kwargs_inplace = {"inplace": inplace} if inplace is not None else {}

        layers = [torch.nn.ConvTranspose2d(num_channels_in, num_channels_out, shape_kernel_conv, **kwargs_conv)]
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
        shape_input,
        nums_channels_hidden_body,
        nums_channels_hidden_head,
        num_channels_out,
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
                BlockConv2d(
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

    def forward(self, input):
        output = self.body(input)
        output = self.head(output)
        return output


class CNN2dEncoder(torch.nn.Module):
    def __init__(self, shape_input, nums_channels_hidden, num_channels_out, kwargs_block):
        super().__init__()

        self.body = None
        self.kwargs_block = kwargs_block or {}
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_out = num_channels_out
        self.shape_input = shape_input

        self._init()

    def _init(self):
        nums_channels_body = [self.shape_input[0]] + list(self.nums_channels_hidden) + [self.num_channels_out]
        modules = []
        for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
            modules.append(
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **self.kwargs_block,
                )
            )
        self.body = torch.nn.Sequential(*modules)

    def forward(self, input):
        output = self.body(input)
        return output


class CNN2dDecoder(torch.nn.Module):
    def __init__(self, shape_input, nums_channels_hidden, num_channels_out, kwargs_block):
        super().__init__()

        self.body = None
        self.kwargs_block = kwargs_block or {}
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_out = num_channels_out
        self.shape_input = shape_input

        self._init()

    def _init(self):
        nums_channels_body = [self.shape_input[0]] + list(self.nums_channels_hidden) + [self.num_channels_out]
        modules = []
        for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
            modules.append(
                BlockDeconv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **self.kwargs_block,
                )
            )
        self.body = torch.nn.Sequential(*modules)

    def forward(self, input):
        output = self.body(input)
        return output


# class CNN2dEncoder(torch.nn.Module):
#     def __init__(
#         self,
#         shape_input,
#         nums_channels_hidden_body,
#         num_channels_out,
#         kwargs_body=None,
#     ):
#         super().__init__()
#         kwargs_body = kwargs_body or {}

#         nums_channels_body = [shape_input[0]] + list(nums_channels_hidden_body)
#         blocks_body = []
#         for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
#             blocks_body.append(
#                 BlockCNN2d(
#                     num_channels_in=num_channels_i,
#                     num_channels_out=num_channels_o,
#                     **kwargs_body,
#                 )
#             )
#         self.body = torch.nn.Sequential(*blocks_body)

#         self.head = torch.nn.Sequential(
#             torch.nn.AdaptiveAvgPool2d(output_size=1),
#             torch.nn.Flatten(start_dim=-3),
#             torch.nn.Linear(in_features=num_channels_o, out_features=num_channels_out),
#         )

#     def forward(self, input):
#         output = self.body(input)
#         output = self.head(output)
#         output = torch.squeeze(output)
#         return output


# class CNN2dEncoderSpatial(torch.nn.Module):
#     def __init__(
#         self,
#         shape_input,
#         nums_channels_hidden_body,
#         num_channels_out,
#         kwargs_body=None,
#     ):
#         super().__init__()
#         kwargs_body = kwargs_body or {}

#         nums_channels_body = [shape_input[0]] + list(nums_channels_hidden_body) + [num_channels_out]
#         blocks_body = []
#         for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
#             blocks_body.append(
#                 BlockCNN2d(
#                     num_channels_in=num_channels_i,
#                     num_channels_out=num_channels_o,
#                     **kwargs_body,
#                 )
#             )
#         self.body = torch.nn.Sequential(*blocks_body)

#     def forward(self, input):
#         output = self.body(input)
#         return output


# class CNN3dResnet(torch.nn.Module):
#     def __init__(
#         self,
#         shape_input,
#         nums_channels_hidden_body,
#         num_channels_out,
#         kwargs_body=None,
#     ):
#         super().__init__()
#         kwargs_body = kwargs_body or {}

#         nums_channels_body = [shape_input[0]] + list(nums_channels_hidden_body) + [num_channels_out]
#         blocks_body = []
#         for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
#             blocks_body.append(
#                 BlockCNN2d(
#                     num_channels_in=num_channels_i,
#                     num_channels_out=num_channels_o,
#                     **kwargs_body,
#                 )
#             )
#         self.body = torch.nn.Sequential(*blocks_body)

#         self.head = torch.nn.AdaptiveAvgPool2d(output_size=1)

#     def forward(self, input):
#         output = self.body(input)
#         output = self.head(output)
#         output = torch.squeeze(output)
#         return output
