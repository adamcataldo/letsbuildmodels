import torch
import unittest
from lbm.models import MLPClassifier

class TestMLPClassifier(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.layers = [20, 10]
        self.classes = 3
        self.means = torch.zeros(self.in_features)
        self.stds = torch.ones(self.in_features)
        self.model = MLPClassifier(self.in_features, self.layers, self.classes, self.means, self.stds)

    def test_forward_shape(self):
        x = torch.randn(5, self.in_features)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (5, self.classes))

    def test_forward_normalization(self):
        x = torch.randn(5, self.in_features)
        normalized_x = self.model.normalizer(x)
        output = self.model.forward(x)
        self.assertTrue(torch.allclose(normalized_x, x, atol=1e-6))

    def test_forward_softmax(self):
        x = torch.randn(5, self.in_features)
        output = self.model.forward(x)
        softmax_sum = torch.sum(output, dim=1)
        self.assertTrue(torch.allclose(softmax_sum, torch.ones(5), atol=1e-6))

    def test_has_parameters(self):
        self.assertTrue(len(list(self.model.parameters())) > 0)

if __name__ == '__main__':
    unittest.main()