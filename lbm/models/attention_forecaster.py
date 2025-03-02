import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, lookback=256, embed_dim=32):
        super().__init__()
        pe = torch.zeros(lookback, embed_dim)
        
        # Position indices: shape [num_prices, 1]
        positions = torch.arange(0, lookback, dtype=torch.float).unsqueeze(1)
        
        # Indices for the even embedding dimensions: shape [embed_dim//2]
        even_dims = torch.arange(0, embed_dim, 2).float()
        
        # 1 / (10000^(2i / embed_dim))) terms
        div_term = torch.exp(
            -torch.log(torch.tensor(10000.0)) * (even_dims / embed_dim)
        )

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + self.positional_encoding
        return x

class AttentionForecaster(nn.Module):
    def __init__(
        self,
        lookback: int = 256,
        num_signals: int = 1,
        embed_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        A single-step forecaster using a Transformer encoder.

        Args:
            lookback (int): Number of past time steps to consider. Default = 256.
            num_signals (int): Number of signals/features at each time step. Default = 1.
            d_model (int): Dimensionality of the embedding. Default = 32.
            nhead (int): Number of attention heads in the Transformer layers. Default = 4.
            num_layers (int): Number of Transformer encoder layers. Default = 2.
            dropout (float): Dropout probability. Default = 0.1.
        """
        super().__init__()
        self.lookback = lookback
        self.num_signals = num_signals
        self.embed_dim = embed_dim

        # 1) Learnable linear projection from (num_signals) -> (d_model)
        self.input_embedding = nn.Linear(num_signals, embed_dim)

        # 2) Positional encoding for the input sequence
        self.positional_embedding = PositionalEmbedding(lookback, embed_dim)

        # 3) Transformer encoder (batch_first=True to keep (batch, seq, d_model) ordering)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Final linear layer for single-step forecast
        # We could do more layers (MLP) here if desired
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one-step forecasting.

        Args:
            x (Tensor): Input time series of shape (batch_size, lookback, num_signals).

        Returns:
            Tensor of shape (batch_size,) with the next-step forecast for each series.
        """
        # x shape: (batch_size, lookback, num_signals)

        # 1) Embed input features to dimension d_model
        #    => (batch_size, lookback, d_model)
        x_emb = self.input_embedding(x)

        # 2) Add positional encoding to the embeddings
        #    => (batch_size, lookback, d_model)
        x_pe = self.positional_embedding(x_emb)

        # 3) Pass embeddings through the Transformer encoder
        #    => (batch_size, lookback, d_model)
        encoded = self.transformer_encoder(x_pe)

        # 4) Aggregate the sequence dimension into a single vector
        #    Example: simple average pooling over the time dimension
        #    => (batch_size, d_model)
        #    (If you want something more sophisticated, see below.)
        encoded_avg = encoded.mean(dim=1)

        # 5) Map the aggregated vector to a single value (per batch element)
        #    => (batch_size, 1)
        out = self.output_layer(encoded_avg)

        # 5) Return shape (batch_size,)
        return out.squeeze(-1)
