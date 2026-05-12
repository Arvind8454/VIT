from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from utils.transforms import build_transforms, build_self_supervised_transforms


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.1
    img_size: int = 224
    seed: int = 42


def _get_dataset(name: str, data_dir: str, train: bool, transform):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
    if name == "tinyimagenet":
        split = "train" if train else "val"
        return datasets.ImageFolder(root=f"{data_dir}/tiny-imagenet-200/{split}", transform=transform)
    raise ValueError(f"Unknown dataset: {name}")


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_transform = build_transforms(train=True, img_size=cfg.img_size)
    test_transform = build_transforms(train=False, img_size=cfg.img_size)

    full_train = _get_dataset(cfg.dataset, cfg.data_dir, train=True, transform=train_transform)
    test_dataset = _get_dataset(cfg.dataset, cfg.data_dir, train=False, transform=test_transform)

    val_size = int(len(full_train) * cfg.val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    class_names = getattr(full_train, "classes", None) or [str(i) for i in range(10)]
    return train_loader, val_loader, test_loader, class_names


class ImagesOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        return image


def get_self_supervised_loader(cfg: DataConfig) -> DataLoader:
    transform = build_self_supervised_transforms(img_size=cfg.img_size)
    dataset = _get_dataset(cfg.dataset, cfg.data_dir, train=True, transform=transform)
    dataset = ImagesOnlyDataset(dataset)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    return loader
