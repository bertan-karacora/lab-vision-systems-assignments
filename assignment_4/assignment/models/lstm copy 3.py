import torch


class CellLSTM(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_hidden, use_bias=True, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.use_bias = use_bias

        self.f = torch.nn.Sequential(
            torch.nn.Linear(num_channels_hidden + num_channels_in, num_channels_hidden, bias=use_bias, device=self.device, dtype=self.dtype),
            torch.nn.Sigmoid(),
        )

        self.i = torch.nn.Sequential(
            torch.nn.Linear(num_channels_hidden + num_channels_in, num_channels_hidden, bias=use_bias, device=self.device, dtype=self.dtype),
            torch.nn.Sigmoid(),
        )

        self.c = torch.nn.Sequential(
            torch.nn.Linear(num_channels_hidden + num_channels_in, num_channels_hidden, bias=use_bias, device=self.device, dtype=self.dtype),
            torch.nn.Tanh(),
        )

        self.o = torch.nn.Sequential(
            torch.nn.Linear(num_channels_hidden + num_channels_in, num_channels_hidden, bias=use_bias, device=self.device, dtype=self.dtype),
            torch.nn.Sigmoid(),
        )

    def forward(self, input, state_hidden=None):
        if state_hidden is None:
            state_hidden = torch.autograd.Variable(input.new_zeros(input.size(0), self.hidden_size))
            state_hidden = (state_hidden, state_hidden)

        input_concat = torch.cat((input, state_hidden), dim=-1)

        state_hidden, state_cell = state_hidden

        gates = self.xh(input) + self.hh(state_hidden)

        cy = self.c(input_concat) * self.c(input_concat) + self.c(input_concat) * self.c(input_concat)

        hy = self.c(input_concat) * torch.tanh(cy)

        return (hy, cy)


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = torch.nn.ModuleList()

        self.rnn_cell_list.append(CellLSTM(self.input_size, self.hidden_size, self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(CellLSTM(self.hidden_size, self.hidden_size, self.bias))

        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = torch.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], (hidden[layer][0], hidden[layer][1]))
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1][0], (hidden[layer][0], hidden[layer][1]))
                    hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out
