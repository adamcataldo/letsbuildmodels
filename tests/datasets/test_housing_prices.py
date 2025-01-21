import unittest
import torch
from letsbuildmodels.datasets.housing_prices import HousingPrices, get_housing_prices

class TestHousingPrices(unittest.TestCase):
    def setUp(self):
        self.dataset = HousingPrices(include_ocean_proximity=True)
        self.dataset_no_ocean = HousingPrices(include_ocean_proximity=False)

    def test_length(self):
        self.assertEqual(len(self.dataset), len(self.dataset.features))
        self.assertEqual(len(self.dataset_no_ocean), len(self.dataset_no_ocean.features))

    def test_getitem(self):
        sample, target = self.dataset[0]
        self.assertEqual(sample.shape[0], self.dataset.features.shape[1])
        self.assertEqual(target.shape[0], 1)
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)

    def test_get_z_score(self):
        means, stds = self.dataset.get_z_score()
        self.assertEqual(means.shape[0], self.dataset.features.shape[1])
        self.assertEqual(stds.shape[0], self.dataset.features.shape[1])
        self.assertIsInstance(means, torch.Tensor)
        self.assertIsInstance(stds, torch.Tensor)

    def test_get_housing_prices(self):
        (train_loader, val_loader, test_loader), (means, stds) = get_housing_prices(include_ocean_proximity=True)
        total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        self.assertEqual(total_samples, len(self.dataset))
        self.assertIsInstance(means, torch.Tensor)
        self.assertIsInstance(stds, torch.Tensor)
        self.assertEqual(means.shape[0], self.dataset.features.shape[1])
        self.assertEqual(stds.shape[0], self.dataset.features.shape[1])


    
if __name__ == '__main__':
    unittest.main()