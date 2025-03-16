import unittest
from lbm.datasets.transformed_dataset import TransformedDataset

class TestTransformedDataset(unittest.TestCase):
    def test_default_transforms(self):
        dataset = [(1, 2), (3, 4)]
        transformed_dataset = TransformedDataset(dataset)
        self.assertEqual(transformed_dataset[0], (1, 2))
        self.assertEqual(transformed_dataset[1], (3, 4))

    def test_custom_transforms(self):
        dataset = [(1, 2), (3, 4)]
        x_transform = lambda x: x * 2
        y_transform = lambda y: y + 1
        transformed_dataset = TransformedDataset(dataset, x_transform, y_transform)
        self.assertEqual(transformed_dataset[0], (2, 3))
        self.assertEqual(transformed_dataset[1], (6, 5))

if __name__ == '__main__':
    unittest.main()
