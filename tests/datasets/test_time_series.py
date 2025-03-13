import unittest
import pandas as pd
import torch
from lbm.datasets.time_series import TimeSeriesDataset, TimeSeriesPreprocessor
import numpy as np

class TestTimeSeriesDataset(unittest.TestCase):
    def setUp(self):
        data = {
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        }
        self.df = pd.DataFrame(data)

    def test_simple_case(self):
        dataset = TimeSeriesDataset(self.df, 3, 1, 0)
        self.assertEqual(len(dataset), 7)
        x, y = dataset[2]
        expected_x = torch.tensor([[3, 8], [4, 7], [5, 6]], dtype=torch.float32)
        expected_y = torch.tensor([[6, 5]], dtype=torch.float32)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_horizon(self):
        dataset = TimeSeriesDataset(self.df, 3, 2, 0)
        self.assertEqual(len(dataset), 6)
        x, y = dataset[2]
        expected_x = torch.tensor([[3, 8], [4, 7], [5, 6]], dtype=torch.float32)
        expected_y = torch.tensor([[6, 5], [7, 4]], dtype=torch.float32)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_alignment(self):
        dataset = TimeSeriesDataset(self.df, 3, 1, 2)
        self.assertEqual(len(dataset), 7)
        x, y = dataset[2]
        expected_x = torch.tensor([[3, 8], [4, 7], [5, 6]], dtype=torch.float32)
        expected_y = torch.tensor([[4, 7], [5, 6], [6, 5]], dtype=torch.float32)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_horizon_and_alignment(self):
        dataset = TimeSeriesDataset(self.df, 3, 2, 1)
        self.assertEqual(len(dataset), 6)
        x, y = dataset[2]
        expected_x = torch.tensor([[3, 8], [4, 7], [5, 6]], dtype=torch.float32)
        expected_y = torch.tensor([[5, 6], [6, 5], [7, 4]], dtype=torch.float32)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

class TestTimeSeriesPreprocessor(unittest.TestCase):
    def setUp(self):
        data = {
            'feature1': np.arange(1, 1004),
            'feature2': np.arange(1003, 0, -1),
        }
        self.df = pd.DataFrame(data)

    def test_get_loaders(self):
        preprocessor = TimeSeriesPreprocessor(self.df, 3, 1, 0)
        train_loader, val_loader, test_loader = preprocessor.get_loaders(2)
        self.assertEqual(len(train_loader), 800 / 2)
        self.assertEqual(len(val_loader), 100 / 2)
        self.assertEqual(len(test_loader), 100 / 2)
        for x, y in train_loader:
            self.assertEqual(x.size(), torch.Size([2, 3, 2]))
            self.assertEqual(y.size(), torch.Size([2, 1, 2]))
            break
        for x, y in val_loader:
            self.assertEqual(x.size(), torch.Size([2, 3, 2]))
            self.assertEqual(y.size(), torch.Size([2, 1, 2]))
            self.assertEqual(x[0, 0, 0], torch.tensor(801))
            break
        for x, y in test_loader:
            self.assertEqual(x.size(), torch.Size([2, 3, 2]))
            self.assertEqual(y.size(), torch.Size([2, 1, 2]))
            self.assertEqual(x[0, 0, 0], torch.tensor(901))
            break

