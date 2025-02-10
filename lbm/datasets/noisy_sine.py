import torch
from torch.utils.data import Dataset, DataLoader

class NoisySine(Dataset):
    def __init__(self, n_points=1000, lookback=4, step = 0.01, noise=0.2):
        self.x = torch.arange(1, n_points + 1, dtype=torch.float32)
        self.y = torch.sin(0.01 * self.x) + noise * torch.randn(n_points)
        self.n_points = n_points
        self.lookback = lookback
        
    def __len__(self):
        return self.n_points - self.lookback
    
    def __getitem__(self, idx):
        return self.y[idx:idx+self.lookback], self.y[idx+self.lookback]
    
class Prepocessor():
    def __init__(self, n_points=1000, lookback=4, step = 0.01, noise=0.2):
        self.train_set = NoisySine(n_points, lookback, step, noise) 
        self.val_set = NoisySine(n_points, lookback, step, noise)
        self.test_set = NoisySine(n_points, lookback, step, noise)

    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader