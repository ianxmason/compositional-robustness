"""
Run the static dataloaders on the training set with normalization removed from the transforms.
From this data calculate per channel mean and std for each dataset.
"""
import os
import sys
import numpy as np
import torch
sys.path.append("../")
from data.data_loaders import get_static_dataloaders

datasets = ["EMNIST", "CIFAR", "FACESCRUB"]
data_root = "<your_data_root_here>"
for dataset in datasets:
    corruption_path = os.path.join(data_root, dataset, "Identity")
    if dataset == "EMNIST":  # 3, 28, 28
        train_classes = list(range(47))
    elif dataset == "CIFAR":  # 3, 32, 32
        train_classes = list(range(10))
    elif dataset == "FACESCRUB":  # 3, 100, 100
        train_classes = list(range(388))
    trn_dl, _, _ = get_static_dataloaders(dataset, corruption_path, train_classes, 256, False, 2, True)

    # Direct copy: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    imgs = None
    for x, y in trn_dl:
        if imgs is None:
            imgs = x.cpu()
        else:
            imgs = torch.cat([imgs, x.cpu()], dim=0)
    imgs = imgs.numpy()
    print(imgs.shape)

    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()
    print(mean_r, mean_g, mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()
    print(std_r, std_g, std_b)


