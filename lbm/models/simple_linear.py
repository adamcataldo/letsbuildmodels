import torch

# A PyTorch model called SimpleLinear, with one input and output
class SimpleLinear(torch.nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)
    