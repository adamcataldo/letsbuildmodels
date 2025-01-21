import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds
    
    def forward(self, x):
        return (x - self.means) / self.stds