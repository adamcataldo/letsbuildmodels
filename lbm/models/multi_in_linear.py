from lbm.models.normalizer import Normalizer
from torch import nn

class MultiInLinear(nn.Module):
    def __init__(self, in_features, means, stds):
        super(MultiInLinear, self).__init__()
        self.normalizer = Normalizer(means, stds)
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        normalized = self.normalizer(x)
        y = self.linear(normalized)
        return y
        