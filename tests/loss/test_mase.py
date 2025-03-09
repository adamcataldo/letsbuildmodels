import torch
import unittest
from lbm.loss import MASELoss
from torch.testing import assert_close


class TestMASELoss(unittest.TestCase):
    def test_zero_loss_when_prediction_equals_target(self):
        batch_size = 2
        seq_length = 5
        feature_dim = 3

        y_true = torch.rand(batch_size, feature_dim)
        y_pred = y_true.clone()  # same as y_true
        x = torch.rand(batch_size, seq_length, feature_dim)

        criterion = MASELoss()
        loss = criterion(y_pred, y_true, x)

        assert_close(loss, torch.zeros_like(loss))

    def test_output_shape(self):
        batch_size = 4
        seq_length = 6
        feature_dim = 2

        y_true = torch.randn(batch_size, feature_dim)
        y_pred = torch.randn(batch_size, feature_dim)
        x = torch.randn(batch_size, seq_length, feature_dim)

        criterion = MASELoss()
        loss = criterion(y_pred, y_true, x)

        assert loss.shape == (feature_dim,), (
            f"Expected output shape ({feature_dim},), got {loss.shape}"
        )

    def test_hand_computed_example(self):
        x = torch.tensor([[[1.0], [2.0], [4.0]]])  # shape (1,3,1)
        y_true = torch.tensor([[1.0]])            # shape (1,1)
        y_pred = torch.tensor([[2.0]])            # shape (1,1)

        criterion = MASELoss(eps=1e-12)
        loss = criterion(y_pred, y_true, x)

        expected_mase = torch.tensor([1.0 / 3.0])  # shape (1,)

        assert_close(loss, expected_mase)

    def test_backpropagation(self):
        batch_size = 3
        seq_length = 4
        feature_dim = 2

        y_true = torch.randn(batch_size, feature_dim, requires_grad=False)
        y_pred = torch.randn(batch_size, feature_dim, requires_grad=True)
        x = torch.randn(batch_size, seq_length, feature_dim, requires_grad=False)

        criterion = MASELoss()
        loss = criterion(y_pred, y_true, x)

        scalar_loss = loss.sum()  
        scalar_loss.backward()

        assert y_pred.grad is not None, "Expected non-None gradient for y_pred."

if __name__ == '__main__':
    unittest.main()