import torch

class MASELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.requires_inputs = True

    def forward(self, y_pred, y_true, x):
        """
        :param y_pred: Predictions with shape (batch_size, feature_dim)
        :param y_true: Ground truth values with shape (batch_size, feature_dim)
        :param x: Previous target values with shape (batch_size, seq_length, feature_dim)
        :return: MASE loss (scalar)
        """
        error = torch.abs(y_true - y_pred)
        consecutive_diff = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        naive_scale = torch.sum(consecutive_diff)
        return torch.mean(error / (naive_scale + self.eps), 0)