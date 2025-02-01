from lbm.models.normalizer import Normalizer
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, in_features, layers, classes, means, stds, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.normalizer = Normalizer(means, stds)
        self.layers = nn.ModuleList()
        prev_in = in_features
        for layer in layers:
            self.layers.append(nn.Linear(prev_in, layer))
            prev_in = layer
        self.layers.append(nn.Linear(prev_in, classes))
        self.dropuut = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.normalizer(x)
        last_layer = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            y = self.relu(layer(y))
            if i < last_layer:
                y = self.dropuut(y)
        p = self.softmax(y)
        return p