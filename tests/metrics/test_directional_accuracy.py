import unittest
import torch
from lbm.metrics import DirectionalAccuracy

class TestDirectionalAccuracy(unittest.TestCase):
    def test_compute_all_correct(self):
        metric = DirectionalAccuracy()
        input_sequences = torch.tensor([[[0.0], [1.0], [2.0]], 
                                        [[1.0], [2.0], [3.0]]]).transpose(1, 0)
        actual_outputs = torch.tensor([[2.0], [3.0], [4.0]])
        predicted_outputs = torch.tensor([[2.5], [3.5], [4.5]])
        metric.update(input_sequences, actual_outputs, predicted_outputs)
        result = metric.compute()
        self.assertEqual(result, 1.0)

    def test_compute_all_incorrect(self):
        metric = DirectionalAccuracy()
        input_sequences = torch.tensor([[[0.0], [1.0], [2.0]], 
                                        [[1.0], [2.0], [3.0]]]).transpose(1, 0)
        actual_outputs = torch.tensor([[2.0], [3.0], [4.0]])
        predicted_outputs = torch.tensor([[0.5], [1.5], [2.5]])
        metric.update(input_sequences, actual_outputs, predicted_outputs)
        result = metric.compute()
        self.assertEqual(result, 0.0)

    def test_compute_mixed(self):
        metric = DirectionalAccuracy()
        input_sequences = torch.tensor([[[0.0], [1.0], [2.0]], 
                                        [[1.0], [2.0], [3.0]]]).transpose(1, 0)
        actual_outputs = torch.tensor([[2.0], [1.0], [4.0]])
        predicted_outputs = torch.tensor([[2.5], [1.5], [2.5]])
        metric.update(input_sequences, actual_outputs, predicted_outputs)
        result = metric.compute()
        self.assertAlmostEqual(result, 2/3)

    def test_sequence_outputs(self):
        metric = DirectionalAccuracy()
        input_sequences = torch.tensor([[[0.0], [1.0], [2.0]], 
                                        [[1.0], [2.0], [3.0]]]).transpose(1, 0)
        actual_outputs = torch.tensor([[[0.1], [0.1], [0.1]],
                                       [[2.0], [1.0], [4.0]]]).transpose(1, 0)
        pred_outputs = torch.tensor([[[5.0], [5.0], [5.0]],
                                     [[2.5], [1.5], [2.5]]]).transpose(1, 0)
        metric.update(input_sequences, actual_outputs, pred_outputs)
        result = metric.compute()
        self.assertAlmostEqual(result, 2/3)


if __name__ == '__main__':
    unittest.main()