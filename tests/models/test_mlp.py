import torch
import unittest
from lbm.models import NormedHiddenLayer, NormedSoftmaxLayer, DenseClassifier

class TestNormedHiddenLayer(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.out_features = 20
        self.layer = NormedHiddenLayer(self.in_features, self.out_features)

    def test_forward_shape(self):
        x = torch.randn(5, self.in_features)
        output = self.layer.forward(x)
        self.assertEqual(output.shape, (5, self.out_features))

class TestNormedSoftmaxLayer(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.out_features = 3
        self.layer = NormedSoftmaxLayer(self.in_features, self.out_features)

    def test_forward_shape(self):
        x = torch.randn(5, self.in_features)
        output = self.layer.forward(x)
        self.assertEqual(output.shape, (5, self.out_features))

    def test_forward_softmax(self):
        x = torch.randn(5, self.in_features)
        output = self.layer.forward(x)
        softmax_sum = torch.sum(output, dim=1)
        self.assertTrue(torch.allclose(softmax_sum, torch.ones(5), atol=1e-6))

class TestDenseClassifier(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.layers = [20, 10]
        self.classes = 3
        self.model = DenseClassifier(self.in_features, self.layers, self.classes)

    def test_forward_shape(self):
        x = torch.randn(5, self.in_features)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (5, self.classes))

    def test_forward_softmax(self):
        x = torch.randn(5, self.in_features)
        output = self.model.forward(x)
        softmax_sum = torch.sum(output, dim=1)
        self.assertTrue(torch.allclose(softmax_sum, torch.ones(5), atol=1e-6))

    def test_has_parameters(self):
        self.assertTrue(len(list(self.model.parameters())) > 0)

if __name__ == '__main__':
    unittest.main()
