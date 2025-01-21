import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StudentPerformance(Dataset):
    def __init__(self):
        path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
        csv_file = f"{path}/Student_Performance.csv"
        df = pd.read_csv(csv_file)
        
        # Convert data to PyTorch tensors
        hours_studied = df['Hours Studied'].values
        grades = df['Performance Index'].values
        self.hours_studied = torch.tensor(hours_studied, dtype=torch.float32).unsqueeze(1)  # Add dimension
        self.grades = torch.tensor(grades, dtype=torch.float32).unsqueeze(1)  # Add dimension
    
    def __len__(self):
        return len(self.hours_studied)  # Number of samples
    
    def __getitem__(self, idx):
        # Return single sample with correct shape
        return self.hours_studied[idx], self.grades[idx]

def get_student_data(batch_size=32, train_split=0.8):
    # Initialize dataset
    dataset = StudentPerformance()

    # Split into training and testing sets
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, test_dataloader