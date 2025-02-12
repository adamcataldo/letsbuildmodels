import torch
import unittest
from lbm.models import Resnet18

class TestResnet18(unittest.TestCase):
    def setUp(self):
        self.model = Resnet18(num_classes=6)
        self.input_tensor = torch.randn(10, 3, 150, 150)  # Example input tensor with batch size 1 and 3 color channels

    def test_forward_output_shape(self):
        output = self.model.forward(self.input_tensor)
        self.assertEqual(output.shape, (10, 6), "Output shape is incorrect")


if __name__ == '__main__':
    unittest.main()