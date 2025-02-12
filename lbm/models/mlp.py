from lbm.models.normalizer import Normalizer
from torch import nn

class NormedHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(NormedHiddenLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.normalizer = nn.BatchNorm1d(out_features)
        self.dropuut = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.linear(x)
        y = self.normalizer(y)
        y = self.dropuut(y)
        y = self.relu(y)
        return y
    
class NormedSoftmaxLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedSoftmaxLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.normalizer = nn.BatchNorm1d(out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.linear(x)
        y = self.normalizer(y)
        y = self.softmax(y)
        return y

class DenseClassifier(nn.Module):
    def __init__(self, in_features, layers, classes, dropout=0.5):
        super(DenseClassifier, self).__init__()
        self.layers = nn.ModuleList()
        prev_in = in_features
        for layer in layers:
            self.layers.append(NormedHiddenLayer(prev_in, layer, dropout))
            prev_in = layer
        self.layers.append(NormedSoftmaxLayer(prev_in, classes))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y