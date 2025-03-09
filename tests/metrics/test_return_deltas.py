import unittest
import torch
import numpy as np
from lbm.metrics.return_deltas import ReturnDeltas

class TestReturnDeltas(unittest.TestCase):
    def setUp(self):
        self.metric = ReturnDeltas()

    def test_initial_state(self):
        self.assertTrue(torch.equal(self.metric.final_inputs, torch.tensor([], device=self.metric.device)))
        self.assertTrue(torch.equal(self.metric.actual_outputs, torch.tensor([], device=self.metric.device)))
        self.assertTrue(torch.equal(self.metric.predicted_outputs, torch.tensor([], device=self.metric.device)))

    def test_update(self):
        input_sequences = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        actual_outputs = torch.tensor([[5.0], [6.0]])
        predicted_outputs = torch.tensor([[7.0], [8.0]])
        self.metric.update(input_sequences, actual_outputs, predicted_outputs)
        
        self.assertTrue(torch.equal(self.metric.final_inputs, torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(self.metric.actual_outputs, torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(self.metric.predicted_outputs, torch.tensor([7.0, 8.0])))

    def test_compute(self):
        input_sequences = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        actual_outputs = torch.tensor([[5.0], [6.0]])
        predicted_outputs = torch.tensor([[7.0], [8.0]])
        self.metric.update(input_sequences, actual_outputs, predicted_outputs)
        
        result = self.metric.compute()
        expected_result = np.array([1.0, 0.5])
        np.testing.assert_almost_equal(result, expected_result)

    def test_merge_state(self):
        metric1 = ReturnDeltas()
        metric2 = ReturnDeltas()
        
        input_sequences1 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        actual_outputs1 = torch.tensor([[5.0], [6.0]])
        predicted_outputs1 = torch.tensor([[7.0], [8.0]])
        metric1.update(input_sequences1, actual_outputs1, predicted_outputs1)
        
        input_sequences2 = torch.tensor([[[9.0], [10.0]], [[11.0], [12.0]]])
        actual_outputs2 = torch.tensor([[13.0], [14.0]])
        predicted_outputs2 = torch.tensor([[15.0], [16.0]])
        metric2.update(input_sequences2, actual_outputs2, predicted_outputs2)
        
        self.metric.merge_state([metric1, metric2])
        
        self.assertTrue(torch.equal(self.metric.final_inputs, torch.tensor([2.0, 4.0, 10.0, 12.0])))
        self.assertTrue(torch.equal(self.metric.actual_outputs, torch.tensor([5.0, 6.0, 13.0, 14.0])))
        self.assertTrue(torch.equal(self.metric.predicted_outputs, torch.tensor([7.0, 8.0, 15.0, 16.0])))

if __name__ == '__main__':
    unittest.main()