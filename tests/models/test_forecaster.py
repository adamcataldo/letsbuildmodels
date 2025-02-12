import torch
import unittest
from lbm.models import Forecaster

class TestForecaster(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.num_layers = 2
        self.batch_size = 5
        self.seq_length = 15
        self.model = Forecaster(self.input_size, self.hidden_size, self.num_layers)
        self.input_data = torch.randn(self.seq_length, self.batch_size, self.input_size)

    def test_forward_shape(self):
        output = self.model(self.input_data)
        self.assertEqual(output.shape, (self.batch_size, self.input_size))

    def test_forward_output(self):
        output = self.model(self.input_data)
        self.assertTrue(torch.is_tensor(output))

if __name__ == '__main__':
    unittest.main()