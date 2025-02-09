from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.LazyLinear(num_classes) 

    def forward(self, x):
        return self.model(x)
