import math
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        # print(f"DEBUG: TemporalCnn - n_inputs={n_inputs}, n_outputs={n_outputs}, kernel_size={kernel_size}")  # Add this
        # print(f"Conv1d created: in_channels={n_inputs}, out_channels={n_outputs}, kernel_size={kernel_size}")
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)

class Tcn_Local(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        # First layer: in_channels=num_inputs, out_channels=num_outputs
        layers.append(TemporalCnn(num_inputs, num_outputs, kernel_size, stride=1, dilation=1,
                                  padding=(kernel_size - 1), dropout=dropout))
        # Next layers: in_channels=num_outputs, out_channels=num_outputs
        for i in range(1, num_levels):
            layers.append(TemporalCnn(num_outputs, num_outputs, kernel_size, stride=1, dilation=1,
                                      padding=(kernel_size - 1), dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Tcn_Global(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2(max(((num_inputs - 1) * (2 - 1) / (kernel_size - 1) + 1), 1)))
        # First layer: in_channels=num_inputs, out_channels=num_outputs
        layers.append(TemporalCnn(num_inputs, num_outputs, kernel_size, stride=1, dilation=1,
                                  padding=(kernel_size - 1), dropout=dropout))
        # Next layers: in_channels=num_outputs, out_channels=num_outputs
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            layers.append(TemporalCnn(num_outputs, num_outputs, kernel_size, stride=1, dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)