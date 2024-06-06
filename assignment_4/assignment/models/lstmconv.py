import torch


class CellLSTMConv2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, use_bias=True, shape_kernel_conv=(5, 5), kwargs_conv=None):
        super().__init__()

        self.kwargs_conv = kwargs_conv or {}
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.shape_kernel_conv = shape_kernel_conv
        self.use_bias = use_bias

        # Parallel execution of convolutional operations of all gates
        self.conv = torch.nn.Conv2d(
            self.num_channels_out + self.num_channels_in,
            4 * self.num_channels_out,
            self.shape_kernel_conv,
            bias=self.use_bias,
            padding="same",
            **self.kwargs_conv,
        )

    def forward(self, input, states=None):
        if states is None:
            zeros = torch.zeros(input.shape[0], self.hidden_size, input.shape[-2], input.shape[-1], dtype=input.dtype, device=input.device)
            states = (zeros, zeros)

        state_hidden, state_cell = states
        input_concat = torch.cat((input, state_hidden), dim=1)

        output_conv = self.conv(input_concat)
        chunks = torch.chunk(output_conv, 4, dim=1)
        f = torch.sigmoid(chunks[0])
        i = torch.sigmoid(chunks[1])
        c = torch.tanh(chunks[2])
        o = torch.sigmoid(chunks[3])

        state_cell = f * state_cell + i * c
        state_hidden = o * torch.tanh(state_cell)

        return state_hidden, state_cell


class LSTMConv2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_layers=1, name_func_init="zeros", use_bias=True, shape_kernel_conv=(5, 5), kwargs_conv=None):
        super().__init__()

        self.cells = None
        self.kwargs_conv = kwargs_conv or {}
        self.name_func_init = name_func_init
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.num_layers = num_layers
        self.shape_kernel_conv = shape_kernel_conv
        self.use_bias = use_bias

        self._init()

    def _init(self):
        cells = []
        for i in range(self.num_layers):
            cells += [
                CellLSTMConv2d(
                    num_channels_in=self.num_channels_in if i == 0 else self.num_channels_out,
                    num_channels_out=self.num_channels_out,
                    use_bias=self.use_bias,
                    shape_kernel_conv=self.shape_kernel_conv,
                    kwargs_conv=self.kwargs_conv,
                )
            ]
        self.cells = torch.nn.ModuleList(cells)

    def forward(self, input):
        states_hidden, states_cell = self._init_states(input)

        outputs_cells = []
        for s in range(input.shape[1]):
            input_cell = input[:, s, ...]
            for c, lstm_cell in enumerate(self.cells):
                states_hidden[c], states_cell[c] = lstm_cell(input_cell, (states_hidden[c], states_cell[c]))
                input_cell = states_hidden[c]
            outputs_cells += [states_hidden[-1]]
        outputs_cells = torch.stack(outputs_cells, dim=1)

        return outputs_cells[:, -1, ...]

    def _init_states(self, input):
        func = getattr(torch, self.name_func_init)
        states_hidden = [
            func(
                input.shape[0],
                self.num_channels_out,
                input.shape[-2],
                input.shape[-1],
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            for _ in range(self.num_layers)
        ]
        states_cell = [
            func(
                input.shape[0],
                self.num_channels_out,
                input.shape[-2],
                input.shape[-1],
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            for _ in range(self.num_layers)
        ]

        return states_hidden, states_cell
