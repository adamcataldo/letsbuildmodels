import torch
import unittest
from lbm.models import NormalizedGRU

class TestNormalizedGRU(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.num_layers = 2
        self.batch_size = 4
        self.seq_len = 5
        self.model = NormalizedGRU(self.input_size, self.hidden_size, 
                                   self.num_layers)
        self.input_tensor = torch.randn(self.seq_len, self.batch_size,
                                        self.input_size)

    def test_forward_output_shape(self):
        output, hidden = self.model.forward(self.input_tensor)
        self.assertEqual(output.shape, (self.seq_len, self.batch_size, 
                                        self.hidden_size))
        self.assertEqual(hidden.shape, 
                         (self.num_layers, self.batch_size, self.hidden_size))

    def test_forward_output_type(self):
        output, hidden = self.model.forward(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(hidden, torch.Tensor)

if __name__ == '__main__':
    unittest.main()