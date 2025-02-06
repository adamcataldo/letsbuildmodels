import unittest
import torch
from lbm.datasets.intel_images import get_loaders, IntelImages
from torch.utils.data import DataLoader

class TestGetLoaders(unittest.TestCase):
    def test_get_loaders(self):
        batch_size = 64
        train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size)
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        for image, label in train_loader:
            self.assertEqual(image.size(), torch.Size((64, 3, 150, 150)))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(label.size(), torch.Size((64, 6)))
            self.assertEqual(label.dtype, torch.int64)
            break

if __name__ == '__main__':
    unittest.main()