from lbm.models.normalizer import Normalizer
from torch import nn

class MultinomialLogistic(nn.Module):
    def __init__(self, in_features, classes, means, stds):
        super(MultinomialLogistic, self).__init__()
        self.normalizer = Normalizer(means, stds)
        self.in_features = in_features
        self.linear = nn.Linear(in_features, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        normalized = self.normalizer(x)
        y = self.linear(normalized)
        p = self.softmax(y)
        return p