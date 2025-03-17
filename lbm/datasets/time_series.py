import torch
from torch.utils.data import Dataset, DataLoader, Subset
from lbm.datasets.transformed_dataset import TransformedDataset

class TimeSeriesDataset(Dataset):
    # df: pandas DataFrame
    def __init__(self, df, lookback, lookahead, tgt_lookback=0):
        self.ts = torch.tensor(df.to_numpy(), dtype=torch.float32)
        self.n_points = self.ts.size()[0]
        self.window = lookback
        self.lookahead = lookahead
        self.tgt_lookback = tgt_lookback

    def __len__(self):
        return self.n_points - self.window - self.lookahead + 1
    
    def __getitem__(self, idx):
        start_x = idx
        end_x = idx + self.window
        start_y = end_x - self.tgt_lookback
        end_y = end_x + self.lookahead
        x = self.ts[start_x:end_x, :]
        y = self.ts[start_y:end_y, :]
        return x, y
    
def last_element(x):
    return x[-1:, :]

class TimeSeriesPreprocessor:
    def __init__(self, df, lookback, lookahead=1, tgt_lookback=0):
        self.dataset = TimeSeriesDataset(df, lookback, lookahead, tgt_lookback)

        # Split the dataset into 80/10/10, in order
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
        self.val_set = TransformedDataset(
            Subset(self.dataset, range(val_start, val_end)),
            y_transform=last_element
        )
        self.test_set = TransformedDataset(
            Subset(self.dataset, range(test_start, test_end)),
            y_transform=last_element
        )

    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(self.train_set, 
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(self.val_set,   batch_size=batch_size)
        test_loader  = DataLoader(self.test_set,  batch_size=batch_size)
        return train_loader, val_loader, test_loader