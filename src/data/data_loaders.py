import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from data.emnist import StaticEMNIST
from data.data_transforms import normalize_0_1


def get_static_emnist_dataloaders(dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tr_ds = StaticEMNIST(dataset_path, keep_classes, which_set='train',
                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    val_ds = StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_multi_static_emnist_dataloaders(dataset_paths, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tr_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='train',
                                        transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
                           for dataset_path in dataset_paths])
    val_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
                            for dataset_path in dataset_paths])
    tst_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='test',
                                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
                            for dataset_path in dataset_paths])

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader
