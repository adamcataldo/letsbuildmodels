import torch

from lbm.metrics import DirectionalAccuracy
from lbm.metrics import ReturnError

def directional_accuracy(model, dataloader,device='cpu'):
    model.to(device)
    model.eval()
    accuracy = DirectionalAccuracy(device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            accuracy.update(inputs, targets, outputs)
    return accuracy.compute()

def return_error(model, dataloader,device='cpu'):
    model.to(device)
    model.eval()
    accuracy = ReturnError(device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            accuracy.update(inputs, targets, outputs)
    return accuracy.compute()