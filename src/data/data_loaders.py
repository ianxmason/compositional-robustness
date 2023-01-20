import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
import torchvision
from torchvision.transforms import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from lib.custom_transforms import SemiRandomCrop, SemiRandomHorizontalFlip
from data.emnist import StaticEMNIST, EMNIST_MEAN, EMNIST_STD
from data.cifar import StaticCIFAR10, CIFAR10_MEAN, CIFAR10_STD
from data.facescrub import StaticFACESCRUB, FACESCRUB_MEAN, FACESCRUB_STD


def create_dataloaders(tr_dataset, val_dataset, tst_dataset, batch_size, shuffle, n_workers, pin_mem):
    """
    Creates dataloaders from datasets
    """
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def create_sampler_dataloaders(tr_dataset, val_dataset, tst_dataset, batch_size, fixed_generator, n_workers, pin_mem):
    """
    Creates dataloaders from datasets. Uses a fixed generator to create the samplers so the data is returned in the
    same order over different corruptions.
    """
    fixed_tr_sampler = RandomSampler(tr_dataset, replacement=False, generator=fixed_generator)
    fixed_val_sampler = RandomSampler(val_dataset, replacement=False, generator=fixed_generator)
    fixed_tst_sampler = RandomSampler(tst_dataset, replacement=False, generator=fixed_generator)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=fixed_tr_sampler,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=fixed_val_sampler,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, sampler=fixed_tst_sampler,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_static_dataloaders(dataset, dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem,
                           fixed_generator=None):
    if dataset == "EMNIST" or dataset == "EMNIST5":
        tr_ds = StaticEMNIST(dataset_path, keep_classes, which_set='train',
                             transform=torchvision.transforms.Compose(
                                 [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
        val_ds = StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                              transform=torchvision.transforms.Compose(
                                  [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
        tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                              transform=torchvision.transforms.Compose(
                                  [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
    elif dataset == "CIFAR":
        tr_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='train',
                              transform=torchvision.transforms.Compose(
                                  [SemiRandomCrop(32, padding=4, seed=1772647822),
                                   SemiRandomHorizontalFlip(seed=1928562283),
                                   ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
        val_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='valid',
                               transform=torchvision.transforms.Compose(
                                   [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
        tst_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='test',
                               transform=torchvision.transforms.Compose(
                                   [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
    elif dataset == "FACESCRUB":
        tr_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='train',
                                transform=torchvision.transforms.Compose(
                                    [SemiRandomCrop(100, padding=10, seed=1772647822),
                                     SemiRandomHorizontalFlip(seed=1928562283),
                                     ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
        val_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='valid',
                                 transform=torchvision.transforms.Compose(
                                     [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
        tst_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='test',
                                 transform=torchvision.transforms.Compose(
                                     [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
    else:
        raise ValueError("Dataset not supported")

    if fixed_generator is None:
        tr_dl, val_dl, tst_dl = create_dataloaders(tr_ds, val_ds, tst_ds, batch_size, shuffle, n_workers, pin_mem)
    else:
        tr_dl, val_dl, tst_dl = create_sampler_dataloaders(tr_ds, val_ds, tst_ds, batch_size, fixed_generator,
                                                           n_workers, pin_mem)

    return tr_dl, val_dl, tst_dl


def get_multi_static_dataloaders(dataset, dataset_paths, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    if dataset == "EMNIST":
        tr_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='train',
                                            transform=torchvision.transforms.Compose(
                                                [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
                               for dataset_path in dataset_paths])
        val_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                                             transform=torchvision.transforms.Compose(
                                                 [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
                                for dataset_path in dataset_paths])
        tst_ds = ConcatDataset([StaticEMNIST(dataset_path, keep_classes, which_set='test',
                                             transform=torchvision.transforms.Compose(
                                                 [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
                                for dataset_path in dataset_paths])
    elif dataset == "CIFAR":
        tr_ds = ConcatDataset([StaticCIFAR10(dataset_path, keep_classes, which_set='train',
                                             transform=torchvision.transforms.Compose(
                                                 [RandomCrop(32, padding=4), RandomHorizontalFlip(),
                                                  ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
                               for dataset_path in dataset_paths])
        val_ds = ConcatDataset([StaticCIFAR10(dataset_path, keep_classes, which_set='valid',
                                              transform=torchvision.transforms.Compose(
                                                  [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
                                for dataset_path in dataset_paths])
        tst_ds = ConcatDataset([StaticCIFAR10(dataset_path, keep_classes, which_set='test',
                                              transform=torchvision.transforms.Compose(
                                                  [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
                                for dataset_path in dataset_paths])
    elif dataset == "FACESCRUB":
        tr_ds = ConcatDataset([StaticFACESCRUB(dataset_path, keep_classes, which_set='train',
                                               transform=torchvision.transforms.Compose(
                                                   [RandomCrop(100, padding=10), RandomHorizontalFlip(),
                                                    ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
                               for dataset_path in dataset_paths])
        val_ds = ConcatDataset([StaticFACESCRUB(dataset_path, keep_classes, which_set='valid',
                                                transform=torchvision.transforms.Compose(
                                                    [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
                                for dataset_path in dataset_paths])
        tst_ds = ConcatDataset([StaticFACESCRUB(dataset_path, keep_classes, which_set='test',
                                                transform=torchvision.transforms.Compose(
                                                    [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
                                for dataset_path in dataset_paths])
    else:
        raise ValueError("Dataset not supported")

    tr_dl, val_dl, tst_dl = create_dataloaders(tr_ds, val_ds, tst_ds, batch_size, shuffle, n_workers, pin_mem)

    return tr_dl, val_dl, tst_dl


def get_transformed_static_dataloaders(dataset, dataset_path, transforms, keep_classes, batch_size, shuffle, n_workers,
                                       pin_mem):
    if dataset == "EMNIST":
        tr_ds = StaticEMNIST(dataset_path, keep_classes, which_set='train',
                             transform=torchvision.transforms.Compose(
                                 transforms + [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
        val_ds = StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                              transform=torchvision.transforms.Compose(
                                  transforms + [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
        tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                              transform=torchvision.transforms.Compose(
                                  transforms + [ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)]))
    elif dataset == "CIFAR":
        tr_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='train',
                              transform=torchvision.transforms.Compose(
                                  transforms + [RandomCrop(32, padding=4), RandomHorizontalFlip(),
                                                ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
        val_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='valid',
                               transform=torchvision.transforms.Compose(
                                   transforms + [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
        tst_ds = StaticCIFAR10(dataset_path, keep_classes, which_set='test',
                               transform=torchvision.transforms.Compose(
                                   transforms + [ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
    elif dataset == "FACESCRUB":
        tr_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='train',
                                transform=torchvision.transforms.Compose(
                                    transforms + [RandomCrop(100, padding=10), RandomHorizontalFlip(),
                                                  ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
        val_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='valid',
                                 transform=torchvision.transforms.Compose(
                                     transforms + [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
        tst_ds = StaticFACESCRUB(dataset_path, keep_classes, which_set='test',
                                 transform=torchvision.transforms.Compose(
                                     transforms + [ToTensor(), Normalize(FACESCRUB_MEAN, FACESCRUB_STD)]))
    else:
        raise ValueError("Dataset not supported")

    tr_dl, val_dl, tst_dl = create_dataloaders(tr_ds, val_ds, tst_ds, batch_size, shuffle, n_workers, pin_mem)

    return tr_dl, val_dl, tst_dl
