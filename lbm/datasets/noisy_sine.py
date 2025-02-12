import torch
from torch.utils.data import Dataset, DataLoader, Subset

class NoisySine(Dataset):
    def __init__(self, n_points=10000, lookback=4, step = 0.01, noise=0.2):
        self.x = torch.arange(1, n_points + 1, dtype=torch.float32)
        self.y = torch.sin(0.01 * self.x) + noise * torch.randn(n_points)
        self.n_points = n_points
        self.lookback = lookback
        
    def __len__(self):
        return self.n_points - self.lookback
    
    def __getitem__(self, idx):
        x = self.y[idx:idx+self.lookback]
        y = self.y[idx+self.lookback]
        return x, y
    
def timeseries_collate_fn(batch):
    """
    Expects a list of (x, y) pairs, where:
      - x is shape (lookback,)
      - y is scalar
    Returns:
      - x of shape (lookback, batch_size, 1)
      - y of shape (batch_size, 1)
    """
    xs, ys = [], []
    for (x, y) in batch:
        # Make x => (lookback, 1), y => (1,)
        xs.append(x.unsqueeze(-1))       # (lookback, 1)
        ys.append(torch.tensor([y]))     # (1,)

    # Now stack across the batch dimension
    # We'll put batch on dimension=1, so final shape is (lookback, batch_size, 1)
    xs = torch.stack(xs, dim=1)  # shape => (lookback, B, 1)
    ys = torch.stack(ys, dim=0)  # shape => (B, 1)

    return xs, ys

class Prepocessor():
    def __init__(self, n_points=10000, lookback=4, step = 0.01, noise=0.2):
        self.dataset = NoisySine(n_points, lookback, step, noise)

        # Split NoisySine into 80/10/10, in order
        length = len(self.dataset)
        n_train = int(0.8 * length)
        n_val   = int(0.1 * length)
        n_test  = length - n_train - n_val
        
        # Indices for each split (chronological order)
        train_start, train_end = 0, n_train
        val_start, val_end     = train_end, train_end + n_val
        test_start, test_end   = val_end,   val_end + n_test
        
        # Subset objects for train/val/test
        self.train_set = Subset(self.dataset, range(train_start, train_end))
        self.val_set   = Subset(self.dataset, range(val_start, val_end))
        self.test_set  = Subset(self.dataset, range(test_start, test_end))
       

    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(
            self.train_set, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=timeseries_collate_fn
        )
        val_loader = DataLoader(
            self.val_set, 
            batch_size=batch_size, 
            collate_fn=timeseries_collate_fn
        )
        test_loader = DataLoader(
            self.test_set, 
            batch_size=batch_size, 
            collate_fn=timeseries_collate_fn
        )
        return train_loader, val_loader, test_loader
