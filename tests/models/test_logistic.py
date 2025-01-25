import unittest
from lbm.models.logistic import Logistic
from lbm.models.normalizer import Normalizer
from torch import nn
import torch

class TestLogisticInit(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.means = torch.tensor([0.5] * self.in_features)
        self.stds = torch.tensor([0.1] * self.in_features)

    def test_call(self):
        model = Logistic(self.in_features, self.means, self.stds)
        sample = torch.randn(1, self.in_features)
        out = model(sample)
        y = out.item()
        self.assertTrue(y >= 0 and y <= 1)

if __name__ == '__main__':
    unittest.main()