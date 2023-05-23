"""
FACESCRUB dataset.

Assumes a flat directory of 50429 images curated as in https://www.pnas.org/doi/epdf/10.1073/pnas.1800901115.
This is turned into and ImageFolder dataset along with methods to generate corrupted versions of the FACESCRUB dataset.
"""
import matplotlib
import argparse
# must select appropriate backend before importing any matplotlib functions
matplotlib.use("Agg")
from matplotlib.image import imsave, imread
from torchvision.datasets import ImageFolder
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

FACESCRUB_MEAN = (0.6166735, 0.46459484, 0.3915861)
FACESCRUB_STD = (0.2678699, 0.2301332, 0.22064506)

class StaticFACESCRUB(ImageFolder):
    def __init__(self, root: str, keep_classes: Any = None, which_set: str = 'train', **kwargs: Any):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected split to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.root = root
        self.which_set = which_set
        self.keep_classes = keep_classes
        super(StaticFACESCRUB, self).__init__(self.split_folder, **kwargs)

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
        classes, class_to_idx = super(StaticFACESCRUB, self)._find_classes(dir)
        classes = self._filter_classes(classes, self.keep_classes)  # filter classes
        classes = [int(c) if c.isdigit() else c for c in classes]  # list of ints and strings, digits and letters
        classes.sort(key=lambda c: ([int, str].index(type(c)), c))  # sorted ints first, then sorted letters
        classes = [str(c) for c in classes]  # back to list of strings

        class_to_idx = {classes[i]: int(classes[i]) for i in range(len(classes))}  # don't re-index classes
        # class_to_idx = {classes[i]: i for i in range(len(classes))}  # re-index classes

        return classes, class_to_idx

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __len__(self):
        return len(self.imgs)


def _get_facescrub_datasets(root_dir):
    print(root_dir)
    sorted_tr_data = _get_sorted_data(os.path.join(root_dir, "FaceScrub"))  # Hardcoded path to flat dir of images
    sorted_tr_data, sorted_val_data, sorted_tst_data = _train_val_tst_split(*sorted_tr_data)

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
            img = transform_fn(img)  # float32 array
        PIL.Image.fromarray(np.uint8(img)).save(img_save_path)  # save as int8 image (grey or colour)
        i += 1


def _create_facescrub_target_datasets(sorted_imgs, sorted_lbls, fns, fn_names, fn_paths):
    for fn, fn_name, fn_path in zip(fns, fn_names, fn_paths):
        print(fn_name)
        mkdir_p(fn_path)
        _create_target_dataset(sorted_imgs, sorted_lbls, fn, fn_path)


def _get_sorted_data(dp):
    if not os.path.exists(os.path.join(dp, "FaceScrub.npz")):
        print("Creating FaceScrub.npz")
        imgs = []
        ids = []
        for img_name in os.listdir(dp):
            img = PIL.Image.open(os.path.join(dp, img_name))
            img = np.array(img)  # uint8, range (0, 255).
            if img.shape == (101, 101, 3):  # Some are 101, 101, 3. In which case we crop to 100, 100, 3.
                img = img[:100, :100, :]
            imgs.append(img)
            ids.append(int(img_name.split("_")[0]) - 100)  # subtract 100 to get 0-indexed labels
        imgs = np.stack(imgs)
        ids = np.array(ids)
        np.savez_compressed(os.path.join(dp, "FaceScrub.npz"), data=imgs, targets=ids)
        print("Created FaceScrub.npz")

    print("Loading FaceScrub.npz")
    facescrub = np.load(os.path.join(dp, "FaceScrub.npz"))
    imgs = facescrub["data"]
    lbls = facescrub["targets"]
    print("Loaded FaceScrub.npz")

    sort_indices = np.argsort(lbls)
    sorted_imgs = imgs[sort_indices]
    sorted_lbls = lbls[sort_indices]

    return sorted_imgs, sorted_lbls


def _train_val_tst_split(sorted_imgs, sorted_lbls):
    # Find out how many images of each class.
    unique_lbls, lbl_counts = np.unique(sorted_lbls, return_counts=True)
    tr_imgs, tr_lbls = [], []
    val_imgs, val_lbls = [], []
    tst_imgs, tst_lbls = [], []

    # Then do an 80/10/10 split.
    for lbl, count in zip(unique_lbls, lbl_counts):
        cls_imgs = sorted_imgs[sorted_lbls == lbl]
        cls_lbls = sorted_lbls[sorted_lbls == lbl]
        ten_pct = int(np.ceil(count * 0.1))

        tr_imgs.append(cls_imgs[:-ten_pct * 2])
        tr_lbls.append(cls_lbls[:-ten_pct * 2])
        val_imgs.append(cls_imgs[-ten_pct * 2:-ten_pct])
        val_lbls.append(cls_lbls[-ten_pct * 2:-ten_pct])
        tst_imgs.append(cls_imgs[-ten_pct:])
        tst_lbls.append(cls_lbls[-ten_pct:])

    tr_imgs = np.concatenate(tr_imgs)
    tr_lbls = np.concatenate(tr_lbls)
    val_imgs = np.concatenate(val_imgs)
    val_lbls = np.concatenate(val_lbls)
    tst_imgs = np.concatenate(tst_imgs)
    tst_lbls = np.concatenate(tst_lbls)

    return (tr_imgs, tr_lbls), (val_imgs, val_lbls), (tst_imgs, tst_lbls)


if __name__ == "__main__":
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
    output_dir = os.path.join(args.data_root, "FACESCRUB/")
    mkdir_p(output_dir)

    if create_datasets:
        # LOAD FACESCRUB DATA
        all_data = _get_facescrub_datasets(args.data_root)
        dset_names = ['train', 'valid', 'test']
        np.random.seed(seed)
        for (sorted_imgs, sorted_labels), dset_name in zip(all_data, dset_names):
            all_corruptions = [['Identity'], ['Contrast'], ['GaussianBlur'], ['ImpulseNoise'], ['Invert'],
                               ['Rotate90'], ['Swirl']]

            c_names = all_corruptions[args.corruption_ID]
            corrs = [getattr(dt, c)() for c in c_names]
            c_name = '-'.join([corr.name for corr in corrs])
            corruption_names = [c_name]
            corruption_paths = [os.path.join(output_dir, c_name, dset_name)]
            corruption_fns = [corrs]

            # Create datasets
            _create_facescrub_target_datasets(sorted_imgs, sorted_labels, corruption_fns, corruption_names,
                                              corruption_paths)
