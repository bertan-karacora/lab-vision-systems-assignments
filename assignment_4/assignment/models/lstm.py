import torch


class CellLSTM(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, use_bias=True):
        super().__init__()

        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.use_bias = use_bias

        # Parallel execution of linear operations of all gates
        self.linear = torch.nn.Linear(self.num_channels_out + self.num_channels_in, 4 * self.num_channels_out, bias=self.use_bias)

    def forward(self, input, states=None):
        if states is None:
            zeros = torch.zeros(input.shape[0], self.num_channels_out, dtype=input.dtype, device=input.device)
            states = (zeros, zeros)

        state_hidden, state_cell = states
        input_concat = torch.cat((input, state_hidden), dim=-1)

        output_linear = self.linear(input_concat)
        chunks = torch.chunk(output_linear, 4, dim=-1)
        f = torch.sigmoid(chunks[0])
        i = torch.sigmoid(chunks[1])
        c = torch.tanh(chunks[2])
        o = torch.sigmoid(chunks[3])

        state_cell = f * state_cell + i * c
        state_hidden = o * torch.tanh(state_cell)

        return state_hidden, state_cell


class LSTMCustom(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_layers=1, name_func_init="zeros", use_bias=True):
        super().__init__()

        self.cells = None
        self.name_func_init = name_func_init
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.num_layers = num_layers
        self.use_bias = use_bias

        self._init()

    def _init(self):
        cells = []
        for i in range(self.num_layers):
            cells += [
                CellLSTM(
                    num_channels_in=self.num_channels_in if i == 0 else self.num_channels_out,
                    num_channels_out=self.num_channels_out,
                    use_bias=self.use_bias,
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
            func(input.shape[0], self.num_channels_out, dtype=input.dtype, layout=input.layout, device=input.device) for _ in range(self.num_layers)
        ]
        states_cell = [func(input.shape[0], self.num_channels_out, dtype=input.dtype, layout=input.layout, device=input.device) for _ in range(self.num_layers)]

        return states_hidden, states_cell


class LSTMCustomClassifier(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_hidden, num_channels_out, num_layers_lstm=1, name_func_init="zeros", use_bias=True):
        super().__init__()

        self.cells = None
        self.head = None
        self.encoder = None
        self.lstm = None
        self.name_func_init = name_func_init
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.num_channels_out = num_channels_out
        self.num_layers_lstm = num_layers_lstm
        self.use_bias = use_bias

        self._init()

    def _init(self):
        self.encoder = torch.nn.Linear(
            in_features=self.num_channels_in,
            out_features=self.num_channels_hidden[0],
        )
        self.lstm = LSTMCustom(
            num_channels_in=self.num_channels_hidden[0],
            num_channels_out=self.num_channels_hidden[1],
            num_layers=self.num_layers_lstm,
            name_func_init=self.name_func_init,
            use_bias=self.use_bias,
        )
        self.classifier = torch.nn.Linear(
            in_features=self.num_channels_hidden[1],
            out_features=self.num_channels_out,
        )

    def forward(self, input):
        print(input.shape)
        output = torch.flatten(input)
        print(output.shape)
        output = self.encoder(input)
        output = self.lstm(output)
        output = self.classifier(output)
        return output
