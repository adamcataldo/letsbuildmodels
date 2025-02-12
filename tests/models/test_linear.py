import unittest
import torch
from lbm.models import Linear

class TestLinear(unittest.TestCase):
    def setUp(self):
        self.in_features = 3
        self.out_features = 2
        self.means = torch.tensor([1.0, 2.0, 3.0])
        self.stds = torch.tensor([0.5, 0.5, 0.5])
        self.model = Linear(self.in_features, self.out_features, self.means, self.stds)

    def test_forward(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = self.model(x)
        self.assertEqual(output.shape, (2, self.out_features))

    def test_normalization(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = self.model.normalizer(x)
        expected_normalized = (x - self.means) / self.stds
        self.assertTrue(torch.allclose(normalized, expected_normalized))

if __name__ == '__main__':
    unittest.main()