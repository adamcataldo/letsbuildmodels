import numpy as np
import torch
from torcheval.metrics import MulticlassAccuracy

# A method to evaluate a PyTorch model on test data
def test(model, dataloader, loss_fn, device='cpu', metrics=[]):
    model.to(device)
    for metric in metrics:
        metric.to(device)
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if hasattr(loss_fn, "requires_inputs") and loss_fn.requires_inputs:
                loss = loss_fn(outputs, targets, inputs)
            else:
                loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            count += inputs.size()[0]
            for metric in metrics:
                metric.update(inputs, targets, outputs)
    return total_loss / count

# A method to train and validate a PyTorch model.
def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, 
                       epochs, device='cpu', metrics=[], patience=3):
    steps = len(train_loader)
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    model.to(device)
    best_loss = float('inf')
    steps_since_best = 0
    for e in range(epochs):
        total_loss = 0
        count = 0
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if hasattr(loss_fn, "requires_inputs") and loss_fn.requires_inputs:
                loss = loss_fn(outputs, targets, inputs)
            else:
                loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l = loss.item()
            total_loss += l
            count += inputs.size()[0]
            print(
                f'Epoch {e+1}/{epochs}, step {i+1}/{steps}         ',
                end='\r'
            )
        epoch_loss = total_loss / count
        train_loss_per_epoch.append(epoch_loss)
        val_loss =  test(model, val_loader, loss_fn, device=device, 
                         metrics=metrics)
        val_loss_per_epoch.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            steps_since_best = 0
        else:
            steps_since_best += 1
        if steps_since_best >= patience:
            break
    return np.array(train_loss_per_epoch), np.array(val_loss_per_epoch)

# A method to convert one-hot encoded targets to class labels (as integers)
def from_onehot(targets):
    return torch.argmax(targets, dim=1)

# A method to evaluate a multiclass accruacy on a PyTorch model on test data
def avg_accuracy(model, dataloader, targets_one_hot=True, device='cpu'):
    model.to(device)
    model.eval()
    accuracy = MulticlassAccuracy()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if targets_one_hot:
                targets = from_onehot(targets)
            accuracy.update(outputs, targets)
    return accuracy.compute().item()
