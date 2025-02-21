from enum import Enum
import torch
import yfinance as yf
from torch.utils.data import Dataset, DataLoader, Subset

class Features(Enum):
    CLOSE_PRICES = 1
    ALL_PRICES = 2
    ALL_PRICES_WITH_ACTIONS = 3

class VOO(Dataset):
    def __init__(self, features=Features.CLOSE_PRICES, lookback=256):
        if features == Features.ALL_PRICES_WITH_ACTIONS:
            d = yf.download('VOO', start='2010-09-09', auto_adjust=False, 
                            actions=True)
            self.x = d[['Close',
                        'Capital Gains',
                        'Dividends',
                        'High',
                        'Low',
                        'Open',
                        'Stock Splits',
                        'Volume']].values
            self.y = d['Adj Close'].values
        elif features == Features.ALL_PRICES:
            d = yf.download('VOO', start='2010-09-09')
            self.x = d[['Close', 'High', 'Low', 'Open', 'Volume']].values
            self.y = d['Close'].values
        else:
            d = yf.download('VOO', start='2010-09-09')
            self.x = d['Close'].values
            self.y = d['Close'].values
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

        self.n_points = len(self.y)
        self.lookback = lookback
        
    def __len__(self):
        return self.n_points - self.lookback
    
    def __getitem__(self, idx):
        x = self.x[idx:idx+self.lookback, :]
        y = self.y[idx+self.lookback, :]
        return x, y
    
def timeseries_collate_fn(batch):
    """
    Expects a list of (x, y) pairs, where:
      - x is shape (lookback,n)
      - y is (1,)
    Returns:
      - x of shape (lookback, batch_size, n)
      - y of shape (batch_size, 1)
    """
    xs, ys = [], []
    for (x, y) in batch:
        xs.append(x)
        ys.append(y)

    xs = torch.stack(xs, dim=1)
    ys = torch.stack(ys, dim=0)

    return xs, ys

class Preprocessor():
    def __init__(self, features=Features.CLOSE_PRICES, lookback=256):
        self.dataset = VOO(features, lookback)

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

    def get_loaders(self, batch_size=64, include_returns=False):
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                 collate_fn=timeseries_collate_fn)
        val_loader   = DataLoader(self.val_set,   batch_size=batch_size,
                                 collate_fn=timeseries_collate_fn)
        test_loader  = DataLoader(self.test_set,  batch_size=batch_size,
                                 collate_fn=timeseries_collate_fn)
        return train_loader, val_loader, test_loader