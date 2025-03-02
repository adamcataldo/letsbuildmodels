import torch
import unittest
from lbm.models.attention_forecaster import AttentionForecaster

class TestAttentionForecaster(unittest.TestCase):
    def setUp(self):
        self.model = AttentionForecaster()
        self.batch_size = 8
        self.lookback = 256
        self.num_signals = 1

    def test_forward_output_shape(self):
        x = torch.randn(self.batch_size, self.lookback, self.num_signals)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (self.batch_size,))

    def test_forward_output_values(self):
        x = torch.randn(self.batch_size, self.lookback, self.num_signals)
        output = self.model.forward(x)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")

    def test_forward_different_input_shapes(self):
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, self.lookback, self.num_signals)
            output = self.model.forward(x)
            self.assertEqual(output.shape, (batch_size,))

if __name__ == '__main__':
    unittest.main()