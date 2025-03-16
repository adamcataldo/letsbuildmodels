from torch.utils.data import Dataset

def identity (x):
    return x

class TransformedDataset(Dataset):
    def __init__(self, dataset, x_transform=identity, y_transform=identity):
        self.dataset = dataset
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.x_transform(x), self.y_transform(y)