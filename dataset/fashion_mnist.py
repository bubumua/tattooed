from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl


class FashionMNIST(pl.LightningDataModule):
    def __init__(self, base_path, batch_size=64, num_workers=1):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download only
        datasets.FashionMNIST(self.base_path / 'data', train=True, download=True, transform=transforms.ToTensor())
        datasets.FashionMNIST(self.base_path / 'data', train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        fashion_mnist_train = datasets.FashionMNIST(self.base_path / 'data', train=True, download=False, transform=transform)
        fashion_mnist_test = datasets.FashionMNIST(self.base_path / 'data', train=False, download=False, transform=transform)

        # train/val split
        fashion_mnist_train, fashion_mnist_val = random_split(fashion_mnist_train, [50000, 10000])

        # assign to use in dataloaders
        self.train_dataset = fashion_mnist_train
        self.val_dataset = fashion_mnist_val
        self.test_dataset = fashion_mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)