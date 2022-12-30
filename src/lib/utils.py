import os
import sys
import errno
import logging
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def seed_torch(seed=404, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Multi-GPU
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Sacrifice speed for exact reproducibility
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def reset_rngs(rng=None, seed=404, deterministic=False):
    if rng is not None:
        rng.seed(seed)
    seed_torch(seed, deterministic)


def generate_batch(data_tuples, dev):
    """
    Takes one or more tuples (x,y) of training data from different data loaders and concatenates them into single
    tensors for training
    """
    for j, data_tuple in enumerate(data_tuples):
        if j == 0:
            x, y = data_tuple[0], data_tuple[1]
        else:
            x_temp, y_temp = data_tuple[0], data_tuple[1]
            x = torch.cat((x, x_temp), dim=0)
            y = torch.cat((y, y_temp), dim=0)
    return x.to(dev), y.to(dev)


def accuracy(outputs, targets):
    _, predictions = torch.max(outputs, 1)
    return 100 * torch.sum(torch.squeeze(predictions).float() == targets).item() / float(targets.size(0))


def visualise_data(images, labels, save_path, title, n_rows=3, n_cols=3):
    # Plot images. Assumes images is numpy array with dtype uint8.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_rows * 2, n_cols * 2))
    for ax, img, lbl in zip(axes.flat, images, labels):
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        ax.imshow(img)
        ax.set_title("Class: {}".format(lbl))
        ax.axis('off')
    fig.suptitle(title)
    plt.savefig(save_path)


def custom_logger(logger_name, level=logging.INFO, stdout=False):
    """
    Method to return a custom logger with the given name and level
    https://stackoverflow.com/questions/54591352/python-logging-new-log-file-each-loop-iteration
    logger_name: path to the logger
    level: logging level
    stdout: if True, log to stdout as well as file
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(message)s")
    # Creating and adding the console handler
    if stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
