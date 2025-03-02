import torch
from torch import nn
from typing import List
from lbm.models.mlp import DenseRegression

class DenseForecaster(nn.Module):
    def __init__(
        self,
        lookback: int = 256,
        num_signals: int = 1,
        hidden_layers: List[int] = [528, 528],
        dropout: float = 0.1
    ):
        super().__init__()
        self.regression = DenseRegression(lookback * num_signals, hidden_layers, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, num_signals)
        x_flat = torch.flatten(x, start_dim=1)
        # x_flat: (batch, lookback * num_signals)
        out = self.regression(x_flat).squeeze(-1)
        # out: (batch,)
        return out.squeeze(-1)
