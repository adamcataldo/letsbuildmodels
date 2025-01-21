import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lbm.workflow import train, test

class TestTrainFunction(unittest.TestCase):
    def setUp(self):
        self.dataloader = [
            (torch.tensor([[1.0, 2.0]]), torch.tensor([1.0])),
            (torch.tensor([[3.0, 4.0]]), torch.tensor([2.0]))
        ]
        self.model = nn.Linear(2, 1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epochs = 2

    def test_train_function(self):
        self.model.return_value = torch.tensor([1.0])
        loss_per_epoch = train(self.model, self.dataloader, self.optimizer, self.loss_fn, self.epochs)

        expected_shape = (2,)
        np.testing.assert_equal(loss_per_epoch.shape, expected_shape)

    def test_test_function(self):
        self.model.return_value = torch.tensor([1.0])
        loss = test(self.model, self.dataloader, self.loss_fn)

        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0)

if __name__ == '__main__':
    unittest.main()