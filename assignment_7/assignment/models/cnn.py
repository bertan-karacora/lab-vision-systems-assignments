import torch

import assignment.libs.utils_import as utils_import


class BlockConv2d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv,
        kwargs_conv=None,
        name_layer_norm=None,
        kwargs_norm=None,
        name_layer_act=None,
        kwargs_act=None,
        name_layer_pool=None,
        kwargs_pool=None,
        prob_dropout=None,
        kwargs_dropout=None,
    ):
        self.kwargs_act = kwargs_act or {}
        self.kwargs_dropout = kwargs_dropout or {}
        self.kwargs_conv = kwargs_conv or {}
        self.kwargs_norm = kwargs_norm or {}
        self.kwargs_pool = kwargs_pool or {}
        self.name_layer_act = name_layer_act
        self.name_layer_norm = name_layer_norm
        self.name_layer_pool = name_layer_pool
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.shape_kernel_conv = shape_kernel_conv
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        modules = [torch.nn.Conv2d(self.num_channels_in, self.num_channels_out, self.shape_kernel_conv, **self.kwargs_conv)]
        if self.name_layer_norm is not None:
            class_module = getattr(torch.nn, self.name_layer_norm)
            if self.name_layer_norm in ["BatchNorm2d", "InstanceNorm2d"]:
                self.kwargs_norm["num_features"] = self.num_channels_out
            modules += [class_module(**self.kwargs_norm)]
        if self.name_layer_act is not None:
            class_module = getattr(torch.nn, self.name_layer_act)
            modules += [class_module(**self.kwargs_act)]
        if self.name_layer_pool is not None:
            class_module = getattr(torch.nn, self.name_layer_pool)
            modules += [class_module(**self.kwargs_pool)]
        if self.prob_dropout is not None:
            modules += [torch.nn.Dropout(self.prob_dropout, **self.kwargs_dropout)]

        super().__init__(*modules)


class BlockConvTranspose2d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv,
        kwargs_conv=None,
        name_layer_norm=None,
        kwargs_norm=None,
        name_layer_act=None,
        kwargs_act=None,
        name_layer_pool=None,
        kwargs_pool=None,
        prob_dropout=None,
        kwargs_dropout=None,
    ):
        self.kwargs_act = kwargs_act or {}
        self.kwargs_dropout = kwargs_dropout or {}
        self.kwargs_conv = kwargs_conv or {}
        self.kwargs_norm = kwargs_norm or {}
        self.kwargs_pool = kwargs_pool or {}
        self.name_layer_act = name_layer_act
        self.name_layer_norm = name_layer_norm
        self.name_layer_pool = name_layer_pool
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.shape_kernel_conv = shape_kernel_conv
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        modules = [torch.nn.ConvTranspose2d(self.num_channels_in, self.num_channels_out, self.shape_kernel_conv, **self.kwargs_conv)]
        if self.name_layer_norm is not None:
            class_module = getattr(torch.nn, self.name_layer_norm)
            if self.name_layer_norm in ["BatchNorm2d", "InstanceNorm2d"]:
                self.kwargs_norm["num_features"] = self.num_channels_out
            modules += [class_module(**self.kwargs_norm)]
        if self.name_layer_act is not None:
            class_module = getattr(torch.nn, self.name_layer_act)
            modules += [class_module(**self.kwargs_act)]
        if self.name_layer_pool is not None:
            class_module = getattr(torch.nn, self.name_layer_pool)
            modules += [class_module(**self.kwargs_pool)]
        if self.prob_dropout is not None:
            modules += [torch.nn.Dropout(self.prob_dropout, **self.kwargs_dropout)]

        super().__init__(*modules)


class EncoderConv2d(torch.nn.Module):
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
            modules += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **self.kwargs_block,
                )
            ]
        self.body = torch.nn.Sequential(*modules)

    def forward(self, input):
        output = self.body(input)
        return output


class DecoderConv2d(torch.nn.Module):
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
            modules += [
                BlockConvTranspose2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **self.kwargs_block,
                )
            ]
        self.body = torch.nn.Sequential(*modules)

    def forward(self, input):
        output = self.body(input)
        return output


# class CNN2d(torch.nn.Module):
#     def __init__(
#         self,
#         shape_input,
#         nums_channels_hidden_body,
#         nums_channels_hidden_head,
#         num_channels_out,
#         kwargs_body=None,
#         kwargs_head=None,
#     ):
#         super().__init__()
#         kwargs_body = kwargs_body or {}
#         kwargs_head = kwargs_head or {}

#         nums_channels_body = [shape_input[0]] + list(nums_channels_hidden_body)
#         blocks_body = []
#         for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
#             blocks_body.append(
#                 BlockConv2d(
#                     num_channels_in=num_channels_i,
#                     num_channels_out=num_channels_o,
#                     **kwargs_body,
#                 )
#             )
#         self.body = torch.nn.Sequential(*blocks_body)

#         with torch.no_grad():
#             dummy_input = torch.unsqueeze(torch.zeros(shape_input), 0)
#             dummy_output_body = self.body(dummy_input)
#             num_channels_in_head = dummy_output_body.nelement()

#         self.head = MLP(
#             num_channels_in=num_channels_in_head,
#             nums_channels_hidden=nums_channels_hidden_head,
#             num_channels_out=num_channels_out,
#             **kwargs_head,
#         )

#     def forward(self, input):
#         output = self.body(input)
#         output = self.head(output)
#         return output
