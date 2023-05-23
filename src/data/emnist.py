"""
EMNIST dataset.

Includes torchvision datasets to load the dataset, along with methods to generate corrupted versions of the EMNIST
dataset.
"""
import matplotlib
import argparse
# must select appropriate backend before importing any matplotlib functions
matplotlib.use("Agg")
from matplotlib.image import imsave, imread
from torchvision.datasets import ImageFolder, EMNIST
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

# These values take black and white images from [0, 1] to [-1, 1]
EMNIST_MEAN = (0.5, 0.5, 0.5)
EMNIST_STD = (0.5, 0.5, 0.5)
# Dataset statistics
# EMNIST_MEAN = (0.17521884, 0.17521884, 0.17521884)
# EMNIST_STD = (0.33335212, 0.33335212, 0.33335212)

class StaticEMNIST(ImageFolder):
    """
    Load batches of images and targets from disk, grouped by class.
    E.g. targets= [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,...], where the num of 0s, 1s, etc. is equal to the number of
    samples per class in that dataset split (len(targets) // num_classes).

    Assumes "balanced split" of EMNIST.
    """

    def __init__(self, root: str, keep_classes: Any = None, which_set: str = 'train', **kwargs: Any):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected split to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.root = root
        self.which_set = which_set
        self.keep_classes = keep_classes
        super(StaticEMNIST, self).__init__(self.split_folder, **kwargs)

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
        classes, class_to_idx = super(StaticEMNIST, self)._find_classes(dir)
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


def _get_emnist_datasets(root_dir):
    print(root_dir)
    sorted_tr_data = _get_sorted_data(EMNIST(root_dir, "balanced", train=True, download=True))
    sorted_tst_data = _get_sorted_data(EMNIST(root_dir, "balanced", train=False, download=True))
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
            img = transform_fn(img)
        PIL.Image.fromarray(np.uint8(img)).save(img_save_path)
        i += 1


def _create_emnist_target_datasets(sorted_imgs, sorted_lbls, fns, fn_names, fn_paths):
    for fn, fn_name, fn_path in zip(fns, fn_names, fn_paths):
        print(fn_name)
        mkdir_p(fn_path)
        _create_target_dataset(sorted_imgs, sorted_lbls, fn, fn_path)


def _get_sorted_data(dp):
    imgs = dp.data.numpy().transpose((0, 2, 1))
    lbls = dp.targets.numpy()

    sort_indices = np.argsort(lbls)
    sorted_imgs = imgs[sort_indices]
    sorted_lbls = lbls[sort_indices]

    return sorted_imgs, sorted_lbls


def _train_val_split(sorted_imgs, sorted_lbls):
    #  Reshape sorted images to have [n_examples_per_class, n_classes, w, h]
    imgs = sorted_imgs.reshape([47, 2400] + list(sorted_imgs.shape[1:]))
    lbls = sorted_lbls.reshape([47, 2400])

    # Use first 2000 examples per class for training
    tr_imgs = imgs[:, :-400].reshape([-1] + list(sorted_imgs.shape[1:]))
    tr_lbls = lbls[:, :-400].reshape([-1])

    # Use last 400 examples per class for validation --> 18,800 examples in total to match the test set size
    val_imgs = imgs[:, -400:].reshape([-1] + list(sorted_imgs.shape[1:]))
    val_lbls = lbls[:, -400:].reshape([-1])

    return (tr_imgs, tr_lbls), (val_imgs, val_lbls)


if __name__ == "__main__":
    # PARAMS
    parser = argparse.ArgumentParser(description='Generate multiple EMNIST corruptions in parallel.')
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    parser.add_argument('--data-root', type=str, default='/om2/user/imason/compositions/datasets/',
                        help="path to directory containing directories of different corruptions")
    args = parser.parse_args()

    create_datasets = True
    # REPRODUCIBILITY
    seed = 1234

    # PATHS
    output_dir = os.path.join(args.data_root, "EMNIST/")

    if create_datasets:
        # LOAD EMNIST DATA
        all_data = _get_emnist_datasets(args.data_root)
        dset_names = ['train', 'valid', 'test']

        # CREATE TARGET DATASETS: E.G. ../datasets/EMNIST/Invert/train/0/21.jpg
        np.random.seed(seed)
        for (sorted_imgs, sorted_labels), dset_name in zip(all_data, dset_names):
            all_corruptions = [['Identity'], ['Contrast'], ['GaussianBlur'], ['ImpulseNoise'], ['Invert'],
                               ['Rotate90'], ['Swirl']]
            # # Can also create all composition using the pkl file
            # with open(os.path.join(output_dir, "corruption_names.pkl"), "rb") as f:
            #     all_corruptions = pickle.load(f)

            c_names = all_corruptions[args.corruption_ID]
            corrs = [getattr(dt, c)() for c in c_names]
            c_name = '-'.join([corr.name for corr in corrs])
            corruption_names = [c_name]
            corruption_paths = [os.path.join(output_dir, c_name, dset_name)]
            corruption_fns = [corrs]

            # Create datasets
            _create_emnist_target_datasets(sorted_imgs, sorted_labels, corruption_fns, corruption_names,
                                           corruption_paths)
