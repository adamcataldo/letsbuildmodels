import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class HousingPrices(Dataset):
    def __init__(self, include_ocean_proximity=False):
        path = kagglehub.dataset_download("camnugent/california-housing-prices")
        csv_file = f"{path}/housing.csv"
        df = pd.read_csv(csv_file)
        
        # Convert data to PyTorch tensors
        features = df.drop('median_house_value', axis=1)
        if include_ocean_proximity:
            # One-hot encode ocean_proximity
            ocean_proximity = pd.get_dummies(features['ocean_proximity'])
            features = pd.concat([features, ocean_proximity], axis=1)
        features = features.drop('ocean_proximity', axis=1)
        features = features.values.astype(float)
        target = df['median_house_value'].values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)  # Add dimension

    def get_z_score(self):
        means = self.features.mean(dim=0)
        stds = self.features.std(dim=0)
        return means, stds
    
    def __len__(self):
        return len(self.features)  # Number of samples
    
    def __getitem__(self, idx):
        # Return single sample with correct shape
        return self.features[idx], self.target[idx]
