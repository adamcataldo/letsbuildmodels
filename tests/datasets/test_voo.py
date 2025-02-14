import unittest
from lbm.datasets.voo import Preprocessor, Features

class TestPrepocessor(unittest.TestCase):
    def test_get_loaders(self):
        batch_size = 32
        lookback = 252
        self.preprocessor = Preprocessor(lookback=lookback)
        train_loader, val_loader, test_loader = self.preprocessor.get_loaders(
            batch_size)
        for x, y in train_loader:
            self.assertEqual(x.size(), (lookback, batch_size, 1)) 
            self.assertEqual(y.size(), (batch_size, 1))
            break
        for x, y in val_loader:
            self.assertEqual(x.size(), (lookback, batch_size, 1)) 
            self.assertEqual(y.size(), (batch_size, 1))
            break
        for x, y in test_loader:
            self.assertEqual(x.size(), (lookback, batch_size, 1)) 
            self.assertEqual(y.size(), (batch_size, 1))
            break

    def test_all_prices(self):
        batch_size = 32
        lookback = 252
        self.preprocessor = Preprocessor(Features.ALL_PRICES, lookback=lookback)
        train_loader, _, _ = self.preprocessor.get_loaders(batch_size)
        for x, y in train_loader:
            self.assertEqual(x.size(), (lookback, batch_size, 5)) 
            self.assertEqual(y.size(), (batch_size, 1))
            break
        
    def test_all_prices_with_actions(self):
        batch_size = 64
        lookback = 252
        self.preprocessor = Preprocessor(Features.ALL_PRICES_WITH_ACTIONS,
                                        lookback=lookback)
        train_loader, _, _ = self.preprocessor.get_loaders(batch_size)
        for x, y in train_loader:
            self.assertEqual(x.size(), (lookback, batch_size, 8)) 
            self.assertEqual(y.size(), (batch_size, 1))
            break



if __name__ == '__main__':
    unittest.main()