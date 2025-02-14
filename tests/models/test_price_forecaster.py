import torch
import unittest
from lbm.models.price_forecaster import PriceForecaster

class TestPriceForecaster(unittest.TestCase):
    def test_forward_shape(self):
        batch_size = 32
        num_prices = 512
        model = PriceForecaster(num_prices=num_prices)
        x = torch.randn(num_prices, batch_size, 1)
        output = model.forward(x)
        self.assertEqual(output.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()