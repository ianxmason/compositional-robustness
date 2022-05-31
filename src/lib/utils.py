import os
import sys
import errno
import logging
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
