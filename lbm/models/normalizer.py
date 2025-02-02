import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, means, stds):
        super(Normalizer, self).__init__()
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
    
    def forward(self, x):
        means = self.means
        stds = self.stds
        return (x - means) / stds