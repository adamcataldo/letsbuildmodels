import unittest
import torch
from lbm.models import SimpleLinear

class TestSimpleLinear(unittest.TestCase):
    def setUp(self):
        self.model = SimpleLinear()
    
    def test_forward(self):
        input_tensor = torch.tensor([[1.0], [2.0], [3.0]])
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([3, 1]))

    def test_parameters(self):
        params = list(self.model.parameters())
        self.assertEqual(len(params), 2)  # weight and bias
        self.assertEqual(params[0].shape, torch.Size([1, 1]))  # weight
        self.assertEqual(params[1].shape, torch.Size([1]))  # bias

if __name__ == '__main__':
    unittest.main()