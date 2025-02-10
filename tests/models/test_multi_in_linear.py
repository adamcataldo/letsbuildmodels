import unittest
import torch
from lbm.models import MultiInLinear, MultiInLinearNormalized

class TestMultiInLinear(unittest.TestCase):
    def setUp(self):
        self.in_features = 3
        self.means = torch.tensor([0.5, 0.5, 0.5])
        self.stds = torch.tensor([0.1, 0.1, 0.1])
        self.model = MultiInLinear(self.in_features, self.means, self.stds)

    def test_forward(self):
        input_tensor = torch.tensor([[0.6, 0.7, 0.8]])
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_normalization(self):
        input_tensor = torch.tensor([[0.6, 0.7, 0.8]])
        normalized_tensor = self.model.normalizer(input_tensor)
        expected_tensor = (input_tensor - self.means) / self.stds
        self.assertTrue(torch.allclose(normalized_tensor, expected_tensor))

class TestMultiInLineaNormalized(unittest.TestCase):
    def setUp(self):
        self.in_features = 3
        self.model = MultiInLinearNormalized(self.in_features)

    def test_forward(self):
        input_tensor = torch.tensor([[0.6, 0.7, 0.8], [0.6, 0.7, 0.8]])
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (2, 1))

if __name__ == '__main__':
    unittest.main()