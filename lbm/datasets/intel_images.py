import kagglehub
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype


import kagglehub

class IntelImages(Dataset):
    def __init__(self, is_train):
        path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        if is_train:
            root_dir = f"{path}/seg_train/seg_train"
        else:
            root_dir = f"{path}/seg_test/seg_test"
        self.images = []
        self.num_classes = 0
        for i, dir in enumerate(os.listdir(root_dir)):
            self.num_classes += 1
            for file in os.listdir(f"{root_dir}/{dir}"):
                image_file = f"{root_dir}/{dir}/{file}"
                label = torch.tensor(i, dtype=torch.long)
                self.images.append((image_file, label))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file, label_number = self.images[idx]
        image = to_dtype(
            decode_image(image_file),
            dtype=torch.float32,
            scale=True
        )
        label = F.one_hot(label_number, num_classes=self.num_classes)
        return image, label
    
def get_loaders(batch_size=64):
    train_dataset = IntelImages(is_train=True)
    val_test_dataset = IntelImages(is_train=False)
    n = len(val_test_dataset) // 2
    val_dataset, test_dataset = random_split(val_test_dataset, [n, n])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader