from lbm.models.normalizer import Normalizer

class MultiInLinear(nn.Module):
    def __init__(self, in_features, means, stds):
        super(MultiInLinear, self).__init__()
        self.normalizer = Normalizer(means, stds)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        normalized = self.normalizer(x)
        return self.linear(normalized)