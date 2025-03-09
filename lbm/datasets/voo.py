from enum import Enum
import torch
import yfinance as yf
from torch.utils.data import Dataset, DataLoader, Subset

class ClosePrices(Dataset):
    def __init__(self, lookback=256):
        d = yf.download('VOO', start='2010-09-09')
        self.x = d['Close'].values
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.n_points = len(self.x)
        self.lookback = lookback
        
    def __len__(self):
        return self.n_points - self.lookback
    
    def __getitem__(self, idx):
        x = self.x[idx:idx+self.lookback, :]
        y = self.x[idx+self.lookback, 0:1]
        return x, y

class ClosePricesPreprocessor():
    def __init__(self, lookback=256):
        self.dataset = ClosePrices(lookback)

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
        train_loader = DataLoader(self.train_set, 
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(self.val_set,   batch_size=batch_size)
        test_loader  = DataLoader(self.test_set,  batch_size=batch_size)
        return train_loader, val_loader, test_loader
    

class Returns(Dataset):
    def __init__(self, lookback=256):
        d = yf.download('VOO', start='2010-09-09')
        self.x = d['Close'].values
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.n_points = len(self.returns)
        self.lookback = lookback
        
    def __len__(self):
        return self.n_points - self.lookback
    
    def __getitem__(self, idx):
        x = self.x[idx:idx+self.lookback, :]
        i = idx+self.lookback
        one = torch.ones((1,1))
        zero = torch.zeros((1,1))
        y =  one if self.x[i, 0] > self.x[i-1, 0] else zero
        return x, y


class ReturnsPreprocessor():
    def __init__(self, lookback=256):
        self.dataset = Returns(lookback)
        self.lookback = lookback

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
        self.prices = self.dataset.x
        self.val_offset = val_start
        self.test_offset = test_start

    def get_loaders(self, batch_size=64, include_returns=False):
        train_loader = DataLoader(self.train_set, batch_size=batch_size)
        val_loader   = DataLoader(self.val_set,   batch_size=batch_size)
        test_loader  = DataLoader(self.test_set,  batch_size=batch_size)
        return train_loader, val_loader, test_loader
    
    def _get_prices(self, batch_index, batch_size, count, offset):
        start = batch_index * batch_size + offset + self.lookback - 1
        end = start + count
        x = self.prices[start:end, 0]
        y = self.prices[start+1:end+1, 0]
        return x, y
    
    def get_val_prices(self, batch_index, batch_size, count):
        return self._get_prices(batch_index, batch_size, count, self.val_offset)
    
    def get_test_prices(self, batch_index, batch_size, count):
        return self._get_prices(batch_index, batch_size, count, 
                                self.test_offset)