import unittest
from unittest.mock import MagicMock
import torch
from torch.utils.data import DataLoader, TensorDataset
from lbm.workflow.price_forecasts import directional_accuracy

class TestDirectionalAccuracy(unittest.TestCase):
    def setUp(self):
        # Create a simple model
        self.model = MagicMock()
        self.model.to = MagicMock()
        self.model.eval = MagicMock()
        self.model.return_value = torch.tensor([[0.5], [0.7], [0.2]])

        # Create a simple dataset
        inputs = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[1], [0], [1]])
        dataset = TensorDataset(inputs, targets)
        self.dataloader = DataLoader(dataset, batch_size=1)

    def test_directional_accuracy(self):
        # Mock DirectionalAccuracy
        mock_accuracy = MagicMock()
        mock_accuracy.compute.return_value = torch.tensor(0.75)
        with unittest.mock.patch(
            'lbm.workflow.price_forecasts.DirectionalAccuracy', 
            return_value=mock_accuracy):
            result = directional_accuracy(self.model, self.dataloader)
            self.assertEqual(result, 0.75)
