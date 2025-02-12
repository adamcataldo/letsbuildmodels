from torch import nn
from lbm.models.normalized_gru import NormalizedGRU

class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = NormalizedGRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        last_value = gru_output[-1, :, :]
        output = self.fc(last_value)
        return output