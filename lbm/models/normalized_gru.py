from torch import nn
import math
import torch

class NormalizedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weight parameters for the "reset" gate
        self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r  = nn.Parameter(torch.Tensor(hidden_size))
        self.r_layer_norm = nn.LayerNorm(hidden_size)

        # Weight parameters for the "update" gate
        self.W_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z  = nn.Parameter(torch.Tensor(hidden_size))
        self.z_layer_norm = nn.LayerNorm(hidden_size)

        # Weight parameters for the candidate hidden
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h  = nn.Parameter(torch.Tensor(hidden_size))
        self.h_layer_norm = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in [self.W_xr, self.W_hr, self.b_r,
                  self.W_xz, self.W_hz, self.b_z,
                  self.W_xh, self.W_hh, self.b_h]:
            nn.init.uniform_(w, -std, std)

    def forward(self, x, h_prev):
        """
        x:      (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        returns h_next: (batch_size, hidden_size)
        """
        # Reset gate
        r_in = self.r_layer_norm(x @ self.W_xr + h_prev @ self.W_hr + self.b_r)
        r_t = torch.sigmoid(r_in)

        # Update gate
        z_in = self.z_layer_norm(x @ self.W_xz + h_prev @ self.W_hz + self.b_z)
        z_t = torch.sigmoid(z_in)

        # Candidate hidden
        h_in = self.h_layer_norm(x @ self.W_xh + (r_t * h_prev) @ self.W_hh + self.b_h)
        h_candidate = torch.tanh(h_in)

        # New hidden state
        h_next = z_t * h_prev + (1 - z_t) * h_candidate
        return h_next

class NormalizedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build a stack of GRU cells
        self.layers = nn.ModuleList([
            NormalizedGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x, h0=None):
        """
        x:  (seq_len, batch_size, input_size)
        h0: (num_layers, batch_size, hidden_size) optional initial hidden state

        Returns:
          output: (seq_len, batch_size, hidden_size)
          hN:     (num_layers, batch_size, hidden_size) final hidden states
        """

        seq_len, batch_size, _ = x.size()

        # If no initial state is provided, set to zeros
        if h0 is None:
            h0 = x.new_zeros((self.num_layers, batch_size, self.hidden_size))

        # We will store the final hidden states for each layer
        hN = []
        
        # The outputs for all time steps from the top layer
        outputs = []

        # Current hidden states (start = h0)
        h_prev = [h0[layer_i] for layer_i in range(self.num_layers)]

        # Process the input sequence one time step at a time
        for t in range(seq_len):
            # Take the t-th slice of the input (batch_size, input_size)
            x_t = x[t]

            # We'll propagate x_t up the stack of GRU cells
            for layer_i, cell in enumerate(self.layers):
                # Compute h_next for this layer
                h_next = cell(x_t, h_prev[layer_i])
                # Update for next time step
                h_prev[layer_i] = h_next
                # The output of this layer is the input to the next layer
                x_t = h_next
            
            # The final x_t after the top layer is the output at time t
            outputs.append(x_t)

        # Collect final hidden states for each layer
        hN = torch.stack(h_prev, dim=0)  # shape (num_layers, batch_size, hidden_size)

        # Stack all time-step outputs into a single tensor
        # shape: (seq_len, batch_size, hidden_size)
        outputs = torch.stack(outputs, dim=0)

        return outputs, hN