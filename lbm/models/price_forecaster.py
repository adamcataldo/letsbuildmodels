import torch
from torch import nn

class SeqFirstLayerNorm(nn.Module):
    def __init__(self, features=1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.layer_norm(x.transpose(0, 1)).transpose(0, 1)

class AddAndNorm(nn.Module):
    def __init__(self, features=1, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = SeqFirstLayerNorm(features)
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.dropout(x)
        return self.layer_norm(x + self.linear(x))
    
class FeedForward(nn.Module):
    def __init__(self, features=1):
        super().__init__()
        self.linear_1 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.layer_norm = SeqFirstLayerNorm(features)
        self.linear_2 = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.linear_2(x)
        return x
    
class FeatureSequenceReducer(nn.Module):
    def __init__(self, features=1, num_prices=256):
        super().__init__()
        self.layer_norm_1 = SeqFirstLayerNorm(features)
        self.linear_1 = nn.Linear(features, 1)
        self.relu = nn.ReLU()
        self.layer_norm_2 = SeqFirstLayerNorm(num_prices)
        self.linear_2 = nn.Linear(num_prices, 1)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.linear_1(x).squeeze(-1)
        x = self.relu(x)
        x = self.layer_norm_2(x).transpose(0, 1)
        x = self.linear_2(x)
        return x

class PriceForecaster(nn.Module):
    def __init__(self, num_prices=256, features=1, dropout=0.1):
        super().__init__()
        self.multi_head = nn.MultiheadAttention(embed_dim=features,
                                                num_heads=features)
        self.add_and_norm_1 = AddAndNorm(features, dropout)
        self.feed_forward = FeedForward(features)
        self.add_and_norm_2 = AddAndNorm(features, dropout)
        self.prediction = FeatureSequenceReducer(features, num_prices)

    def forward(self, x):
        # x is shape (lookback, batch_size, num_prices)
        attn_output, _ = self.multi_head(x, x, x)
        x = self.add_and_norm_1(attn_output + x)
        ff_output = self.feed_forward(x)
        x = self.add_and_norm_2(ff_output + x)
        x = self.prediction(x)
        return x
