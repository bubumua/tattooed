from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
from .gtsrb_dataset import GTSRB_dataset

class GTSRB(pl.LightningDataModule):
    def __init__(self, base_path, batch_size=64, num_workers=1):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])
        GTSRB_dataset(self.base_path / 'data', train=True,  transform=transform)
        GTSRB_dataset(self.base_path / 'data', train=True,  transform=transform)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])
        gtsrb_train = GTSRB_dataset(self.base_path / 'data', train=True,  transform=transform)
        gtsrb_test = GTSRB_dataset(self.base_path / 'data', train=True,  transform=transform)

        # train/val split
        train_size = int(len(gtsrb_train) * 0.8)
        gtsrb_train, gtsrb_val = random_split(gtsrb_train, [train_size, len(gtsrb_train) - train_size])

        # assign to use in dataloaders
        self.train_dataset = gtsrb_train
        self.val_dataset = gtsrb_val
        self.test_dataset = gtsrb_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

