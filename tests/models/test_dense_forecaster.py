import unittest
import torch
from lbm.models import DenseForecaster

class TestDenseForecaster(unittest.TestCase):
    def setUp(self):
        self.lookback = 256
        self.num_signals = 1
        self.hidden_layers = [528, 528]
        self.dropout = 0.1
        self.model = DenseForecaster(
            lookback=self.lookback,
            num_signals=self.num_signals,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        )

    def test_forward_shape(self):
        batch_size = 10
        x = torch.randn(batch_size, self.lookback, self.num_signals)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size,))

    def test_forward_values(self):
        batch_size = 10
        x = torch.randn(batch_size, self.lookback, self.num_signals)
        output = self.model(x)
        self.assertTrue(torch.is_tensor(output))
        self.assertFalse(torch.isnan(output).any())

if __name__ == '__main__':
    unittest.main()