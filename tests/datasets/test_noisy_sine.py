import unittest
import torch
from lbm.datasets.noisy_sine import NoisySine

class TestNoisySine(unittest.TestCase):
    def test_init_default(self):
        dataset = NoisySine()
        self.assertEqual(len(dataset.x), 1000)
        self.assertEqual(len(dataset.y), 1000)
        self.assertEqual(dataset.n_points, 1000)
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

if __name__ == '__main__':
    unittest.main()