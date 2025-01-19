import numpy as np

# A method to train a PyTorch model.
def train(model, dataloader, optimizer, loss_fn, epochs):
    steps = len(dataloader)
    loss_per_epoch = []
    model.train()
    for e in range(epochs):
        total_loss = 0
        count = 0
        for i, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
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
        loss_per_epoch.append(total_loss / count)
    return np.array(loss_per_epoch)