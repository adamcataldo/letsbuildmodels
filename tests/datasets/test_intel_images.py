import unittest
import torch
from lbm.datasets.intel_images import Prepocessor
from torch.utils.data import DataLoader
from itertools import chain

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = Prepocessor()       

    def test_get_loaders(self):
        batch_size = 64
        train_loader, val_loader, test_loader = self.processor.get_loaders(batch_size=batch_size)
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        non_full_batches = 0
        for image, label in chain(train_loader, val_loader, test_loader):
            self.assertEqual(image.size()[1:], torch.Size((3, 150, 150)))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.int64)
            self.assertEqual(image.size()[0], label.size()[0])
            if image.size()[0] < batch_size:
                non_full_batches += 1

        self.assertLessEqual(non_full_batches, 3)

    def test_class_name(self):
        names = ['forest', 'buildings', 'glacier', 'street', 'mountain', 'sea']
        self.assertEqual(self.processor.class_names, names)
        one_hot = torch.tensor(2)
        self.assertEqual(self.processor.class_name(one_hot), 'glacier')

if __name__ == '__main__':
    unittest.main()