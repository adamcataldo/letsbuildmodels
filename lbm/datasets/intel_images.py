import kagglehub
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype, pad


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
        self.class_names = []
        for i, dir in enumerate(os.listdir(root_dir)):
            self.num_classes += 1
            self.class_names.append(dir)
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
        if image.size()[1:] != torch.Size([150, 150]):
            height, width  = image.size()[1:]
            left = (150 - width) // 2 if width < 150 else 0
            right = 150 - width - left if width < 150 else 0
            top = (150 - height) // 2 if height < 150 else 0
            bottom = 150 - height - top if height < 150 else 0
            image = pad(image, (left, top, right, bottom))
        label = label_number
        return image, label


class Prepocessor():
    def __init__(self):
        self.train_dataset = IntelImages(is_train=True)
        val_test_dataset = IntelImages(is_train=False)
        n = len(val_test_dataset) // 2
        self.val_dataset, self.test_dataset = random_split(val_test_dataset, [n, n])
        self.class_names = self.train_dataset.class_names
        
    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def class_name(self, label_tensor):
        return self.class_names[label_tensor.item()]