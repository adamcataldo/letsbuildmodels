from lbm.models.normalizer import Normalizer
from torch import nn

class Logistic(nn.Module):
    def __init__(self, in_features, means, stds):
        super(Logistic, self).__init__()
        self.normalizer = Normalizer(means, stds)
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        normalized = self.normalizer(x)
        y = self.linear(normalized)
        p = self.sigmoid(y)
        return p