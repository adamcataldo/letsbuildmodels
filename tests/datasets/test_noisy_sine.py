import unittest
import torch
from lbm.datasets.noisy_sine import NoisySine, Prepocessor

class TestNoisySine(unittest.TestCase):
    def test_init_default(self):
        dataset = NoisySine()
        self.assertEqual(len(dataset.x), 10000)
        self.assertEqual(len(dataset.y), 10000)
        self.assertEqual(dataset.n_points, 10000)
        self.assertEqual(dataset.lookback, 4)
        self.assertTrue(torch.is_tensor(dataset.x))
        self.assertTrue(torch.is_tensor(dataset.y))

    def test_init_custom(self):
        n_points = 500
        lookback = 10
        step = 0.02
        noise = 0.1
        dataset = NoisySine(n_points=n_points, lookback=lookback, step=step, noise=noise)
        self.assertEqual(len(dataset.x), n_points)
        self.assertEqual(len(dataset.y), n_points)
        self.assertEqual(dataset.n_points, n_points)
        self.assertEqual(dataset.lookback, lookback)
        self.assertTrue(torch.is_tensor(dataset.x))
        self.assertTrue(torch.is_tensor(dataset.y))

class TestPreprocessor(unittest.TestCase):
    def test_sizes(self):
        seq_length = 7
        batch_size = 32
        input_size = 1
        hidden_size = 1
        dataset = Prepocessor(lookback=seq_length)
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=batch_size
        )
        for x, y in train_loader:
            self.assertEqual(x.size(), (seq_length, batch_size, input_size)) 
            self.assertEqual(y.size(), (batch_size, hidden_size))
            break
        for x, y in val_loader:
            self.assertEqual(x.size(), (seq_length, batch_size, input_size)) 
            self.assertEqual(y.size(), (batch_size, hidden_size))
            break
        for x, y in test_loader:
            self.assertEqual(x.size(), (seq_length, batch_size, input_size)) 
            self.assertEqual(y.size(), (batch_size, hidden_size))
            break

    def test_can_iterate(self):
        seq_length = 7
        batch_size = 32
        dataset = Prepocessor(lookback=seq_length)
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=batch_size
        )
        for _, _ in train_loader:
            pass
        for _, _ in val_loader:
            pass
        for _, _ in test_loader: # Test fails here
            pass



if __name__ == '__main__':
    unittest.main()