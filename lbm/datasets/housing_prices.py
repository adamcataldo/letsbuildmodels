from enum import Enum

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class HousingPrices(Dataset):
    def __init__(
            self,
            include_ocean_proximity=False,
            label_cols=['median_house_value']
            ):
        path = kagglehub.dataset_download("camnugent/california-housing-prices")
        csv_file = f"{path}/housing.csv"
        df = pd.read_csv(csv_file)
        
        # Convert data to PyTorch tensors
        features = df.drop(columns=label_cols)
        if include_ocean_proximity:
            # One-hot encode ocean_proximity
            ocean_proximity = pd.get_dummies(features['ocean_proximity'])
            features = pd.concat([features, ocean_proximity], axis=1)
        features = features.drop('ocean_proximity', axis=1)
        self.feature_names = features.columns
        features = features.dropna()
        features = features.values.astype(float)
        target = df[label_cols].values.astype(float)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def get_z_score(self):
        means = self.features.mean(dim=0)
        stds = self.features.std(dim=0)
        return means, stds
    
    def get_feature_names(self):
        return self.feature_names
    
    def __len__(self):
        return len(self.features)  # Number of samples
    
    def __getitem__(self, idx):
        # Return single sample with correct shape
        return self.features[idx], self.target[idx]

class IncomeAndPrices(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data, label = self.original_dataset[idx]
        new_data = torch.cat((data[:, :7], data[:, 8:]), dim=1)
        new_label = torch.cat((data[:, 8:9], label), dim=1)
        return label, data

class Preprocessor:
    def __init__(self,
                 include_ocean_proximity=False,
                 batch_size=32,
                 train_pct=0.8,
                 val_pct=0.1,
                 label_cols=['median_house_value'],
                 ):
        # Initialize dataset
        dataset = HousingPrices(
            include_ocean_proximity=include_ocean_proximity,
            label_cols=label_cols
        )

        # Split into training and testing sets
        n = len(dataset)
        train_size = int(train_pct * n)
        val_size = int(val_pct * n)
        test_size = n - train_size - val_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=batch_size)
        self.test_loader = DataLoader(test, batch_size=batch_size)

        self.means, self.stds = dataset.get_z_score()
        self.feature_names = dataset.get_feature_names()

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_z_score(self):
        return self.means, self.stds
    
    def get_feature_names(self):
        return self.feature_names

