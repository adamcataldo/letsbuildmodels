import torch
import unittest
from lbm.models import Normalizer

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.means = torch.tensor([0.5, 0.5, 0.5])
        self.stds = torch.tensor([0.2, 0.2, 0.2])
        self.normalizer = Normalizer(self.means, self.stds)
    
    def test_forward(self):
        x = torch.tensor([1.0, 1.0, 1.0])
        expected_output = (x - self.means) / self.stds
        output = self.normalizer(x)
        self.assertTrue(torch.allclose(output, expected_output))

if __name__ == '__main__':
    unittest.main()