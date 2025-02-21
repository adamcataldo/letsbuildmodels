import unittest
import torch
from lbm.metrics import ReturnError

class TestDirectionalAccuracy(unittest.TestCase):
    def test_computec_all_corret(self):
        metric = ReturnError()
        input_sequences = torch.tensor([[[0.0], [1.0], [2.0]], 
                                        [[1.0], [2.0], [3.0]]])
        actual_outputs = torch.tensor([[2.0], [4.0], [6.0]])
        predicted_outputs = torch.tensor([[1.5], [5.0], [4.5]])
        metric.update(input_sequences, actual_outputs, predicted_outputs)
        result = metric.compute()
        self.assertAlmostEqual(result, 0.5)

if __name__ == '__main__':
    unittest.main()