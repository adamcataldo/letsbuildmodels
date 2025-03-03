import unittest
from lbm.datasets.voo import ClosePricesPreprocessor

class TestPrepocessor(unittest.TestCase):
    def test_get_loaders(self):
        batch_size = 32
        lookback = 252
        self.preprocessor = ClosePricesPreprocessor(lookback=lookback)
        train_loader, val_loader, test_loader = self.preprocessor.get_loaders(
            batch_size)
        for x, y in train_loader:
            self.assertEqual(x.size(), (batch_size, lookback, 1)) 
            self.assertEqual(y.size(), (batch_size,))
            break
        for x, y in val_loader:
            self.assertEqual(x.size(), (batch_size, lookback, 1)) 
            self.assertEqual(y.size(), (batch_size,))
            break
        for x, y in test_loader:
            self.assertEqual(x.size(), (batch_size, lookback, 1)) 
            self.assertEqual(y.size(), (batch_size,))
            break

if __name__ == '__main__':
    unittest.main()