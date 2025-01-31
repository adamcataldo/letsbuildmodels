import unittest
import torch
from torch.utils.data import DataLoader
from lbm.datasets.housing_locations import HousingLocations, Preprocessor

class TestHousingLocations(unittest.TestCase):
    def setUp(self):
        self.dataset = HousingLocations(include_all_labels=True)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), len(self.dataset.features))

    def test_dataset_item(self):
        features, target = self.dataset[0]
        self.assertEqual(features.shape[0], 2)  # latitude and longitude
        self.assertEqual(target.shape[0], 5)  # one-hot encoded ocean proximity

    def test_get_z_score(self):
        means, stds = self.dataset.get_z_score()
        self.assertEqual(means.shape[0], 2)
        self.assertEqual(stds.shape[0], 2)

    def test_get_feature_names(self):
        feature_names = self.dataset.get_feature_names()
        self.assertEqual(len(feature_names), 2)
        self.assertIn('latitude', feature_names)
        self.assertIn('longitude', feature_names)

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor(include_all_labels=True, batch_size=32, train_pct=0.8, val_pct=0.1)

    def test_data_loaders(self):
        train_loader, val_loader, test_loader = self.preprocessor.get_loaders()
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

    def test_z_score(self):
        means, stds = self.preprocessor.get_z_score()
        self.assertEqual(means.shape[0], 2)
        self.assertEqual(stds.shape[0], 2)

    def test_feature_names(self):
        feature_names = self.preprocessor.get_feature_names()
        self.assertEqual(len(feature_names), 2)
        self.assertIn('latitude', feature_names)
        self.assertIn('longitude', feature_names)

    def test_single_label_constructable(self):
        preprocessor = Preprocessor()

    def test_label_names(self):
        label_names = self.preprocessor.get_label_names()
        self.assertEqual(len(label_names), 5)
        self.assertIn('INLAND', label_names)
        self.assertIn('NEAR BAY', label_names)
        self.assertIn('NEAR OCEAN', label_names)
        self.assertIn('ISLAND', label_names)
        self.assertIn('NEAR OCEAN', label_names)

if __name__ == '__main__':
    unittest.main()