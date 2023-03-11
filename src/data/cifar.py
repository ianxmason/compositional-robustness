"""
CIFAR dataset.

Includes torchvision datasets to load the dataset, along with methods to generate corrupted versions of the CIFAR
dataset.
"""
import matplotlib
import argparse
# must select appropriate backend before importing any matplotlib functions
matplotlib.use("Agg")
from matplotlib.image import imsave, imread
from torchvision.datasets import ImageFolder, CIFAR10
from typing import Any
import PIL.Image
import os
import numpy as np
import torch
import sys
import pickle
sys.path.append("../")
import data.data_transforms as dt
from lib.utils import mkdir_p

CIFAR10_MEAN = (0.49086496, 0.48182634, 0.44628558)
CIFAR10_STD = (0.24671966, 0.24326433, 0.26157698)

class StaticCIFAR10(ImageFolder):
    def __init__(self, root: str, keep_classes: Any = None, which_set: str = 'train', **kwargs: Any):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected split to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.root = root
        self.which_set = which_set
        self.keep_classes = keep_classes
        super(StaticCIFAR10, self).__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.which_set)

    @staticmethod
    def _filter_classes(classes, keep_classes):
        if keep_classes is not None:
            str_keep_classes = [str(c) for c in keep_classes]
            classes = [c for c in classes if c in str_keep_classes]
        return classes

    def _find_classes(self, dir):
        """
        Reorder classes:
            - [0, 1, 10, 100, ..., A, B, ..., a, b, ...] --> [0, 1, 2, 3, ..., A, B, ..., a, b, ...]
        """
        classes, class_to_idx = super(StaticCIFAR10, self)._find_classes(dir)
        classes = self._filter_classes(classes, self.keep_classes)   # filter classes
        classes = [int(c) if c.isdigit() else c for c in classes]    # list of ints and strings, digits and letters
        classes.sort(key=lambda c: ([int, str].index(type(c)), c))   # sorted ints first, then sorted letters
        classes = [str(c) for c in classes]                          # back to list of strings

        class_to_idx = {classes[i]: int(classes[i]) for i in range(len(classes))}  # don't re-index classes
        # class_to_idx = {classes[i]: i for i in range(len(classes))}  # re-index classes

        return classes, class_to_idx

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __len__(self):
        return len(self.imgs)


def _get_cifar_datasets(root_dir):
    print(root_dir)
    sorted_tr_data = _get_sorted_data(CIFAR10(root_dir, train=True, download=True))
    sorted_tst_data = _get_sorted_data(CIFAR10(root_dir, train=False, download=True))
    sorted_tr_data, sorted_val_data = _train_val_split(*sorted_tr_data)

    return [sorted_tr_data, sorted_val_data, sorted_tst_data]


def _create_target_dataset(sorted_imgs, sorted_labels, transform_fns, save_path):
    """ Save grayscale images with PIL.Image as it saves memory with **no information loss**.
    Notes:
        1) ImageFolder dataset automatically converts to RGB.
        2) To regain to pleasant-looking RGBA image of matplotlib's Image.imsave function, simply save
        a single channel of the RGB-converted image. That is, matplotlib's Image.imsave function
        automatically saves grayscale arrays as RGBA arrays (same info, different numbers).
    """
    current_lbl = "Init"
    i = -1
    for img, lbl in zip(sorted_imgs, sorted_labels):
        if current_lbl != lbl:
            i = 0
            current_lbl = lbl
            mkdir_p(os.path.join(save_path, str(current_lbl)))
        img_save_path = os.path.join(save_path, str(current_lbl), str(i) + ".png")
        for transform_fn in transform_fns:
            img = transform_fn(img)                                 # float32 array
        PIL.Image.fromarray(np.uint8(img)).save(img_save_path)      # save as int8 image (grey or colour)
        i += 1


def _create_cifar_target_datasets(sorted_imgs, sorted_lbls, fns, fn_names, fn_paths):
    for fn, fn_name, fn_path in zip(fns, fn_names, fn_paths):
        print(fn_name)
        mkdir_p(fn_path)
        _create_target_dataset(sorted_imgs, sorted_lbls, fn, fn_path)


def _get_sorted_data(dp):
    imgs = dp.data
    lbls = np.array(dp.targets)

    sort_indices = np.argsort(lbls)

    sorted_imgs = imgs[sort_indices]
    sorted_lbls = lbls[sort_indices]

    return sorted_imgs, sorted_lbls


def _train_val_split(sorted_imgs, sorted_lbls):
    #  Reshape sorted images to have [n_examples_per_class, n_classes, w, h]
    imgs = sorted_imgs.reshape([10, 5000] + list(sorted_imgs.shape[1:]))
    lbls = sorted_lbls.reshape([10, 5000])

    # Use first 4000 examples per class for training
    tr_imgs = imgs[:, :-1000].reshape([-1] + list(sorted_imgs.shape[1:]))
    tr_lbls = lbls[:, :-1000].reshape([-1])

    # Use last 1000 examples per class for validation --> 10000 examples in total to match the test set size
    val_imgs = imgs[:, -1000:].reshape([-1] + list(sorted_imgs.shape[1:]))
    val_lbls = lbls[:, -1000:].reshape([-1])

    return (tr_imgs, tr_lbls), (val_imgs, val_lbls)


if __name__ == "__main__":
    # To check if it worked can count the files in each corruption directory and check they are the same == 60034
    # https://stackoverflow.com/questions/15216370/how-to-count-number-of-files-in-each-directory
    # PARAMS
    parser = argparse.ArgumentParser(description='Generate multiple corruptions in parallel.')
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    args = parser.parse_args()

    create_datasets = True
    # REPRODUCIBILITY
    seed = 1234

    # PATHS
    output_dir = os.path.join(args.data_root, "CIFAR/")
    mkdir_p(output_dir)

    if create_datasets:
        # LOAD CIFAR DATA
        all_data = _get_cifar_datasets(args.data_root)
        dset_names = ['train', 'valid', 'test']
        np.random.seed(seed)
        for (sorted_imgs, sorted_labels), dset_name in zip(all_data, dset_names):
            # For CIFAR we only need the elemental corruptions
            all_corruptions = [['Identity'], ['Contrast'], ['GaussianBlur'], ['ImpulseNoise'], ['Invert'],
                               ['Rotate90'], ['Swirl']]

            c_names = all_corruptions[args.corruption_ID]
            corrs = [getattr(dt, c)() for c in c_names]
            c_name = '-'.join([corr.name for corr in corrs])
            corruption_names = [c_name]
            corruption_paths = [os.path.join(output_dir, c_name, dset_name)]
            corruption_fns = [corrs]

            # Create datasets
            _create_cifar_target_datasets(sorted_imgs, sorted_labels, corruption_fns, corruption_names,
                                          corruption_paths)
