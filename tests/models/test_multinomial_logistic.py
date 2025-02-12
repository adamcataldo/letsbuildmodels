import torch
import unittest
from lbm.models import MultinomialLogistic

class TestMultinomialLogistic(unittest.TestCase):
    def setUp(self):
        self.in_features = 4
        self.classes = 3
        self.means = torch.tensor([0.5, 0.5, 0.5, 0.5])
        self.stds = torch.tensor([0.5, 0.5, 0.5, 0.5])
        self.model = MultinomialLogistic(self.in_features, self.classes, self.means, self.stds)

    def test_forward_output_shape(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = self.model.forward(x)
        self.assertEqual(output.shape, (1, self.classes))

    def test_forward_output_sum(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = self.model.forward(x)
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.tensor([1.0])))

    def test_forward_output_non_negative(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = self.model.forward(x)
        self.assertTrue(torch.all(output >= 0))

if __name__ == '__main__':
    unittest.main()