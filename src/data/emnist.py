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


class DynamicEMNIST(EMNIST):
    """
    Load entire dataset into RAM as torch tensors.
    Just like standard EMNIST dataset, but with the option of normalizing / adding colour channels directly
    to torch tensors, rather than via PIL images (slower).
    Pros:
        Avoid per image operations like:
            - casting to and from PIL images
            - normalization
            - converting to colour array
    Cons:
        Inconsistent transforms -- no longer act on a PIL Image, but rather a tensor.
    """
    def __init__(self, root, split="balanced", norm=True, color=True, which_set="train", keep_classes=None,
                 **kwargs):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected split to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.keep_classes = keep_classes
        self.norm = norm
        self.color = color
        self.which_set = which_set
        self.train = (self.which_set in ["train", "valid"])
        super(DynamicEMNIST, self).__init__(root=root, split=split, train=self.train, **kwargs)
        self._prep_set()
        self._norm_data()
        self._color_channels()

    @property
    def raw_folder(self):    # datasets/EMNIST/processed/...
        return os.path.join(self.root, "EMNIST", 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, "EMNIST", 'processed')

    def _norm_data(self):
        if self.norm:
            self.data = self.data.type(torch.FloatTensor) * float(2. / 255.) - 1.  # (0, 255) --> (-1, 1)

    def _color_channels(self):
        if self.color:
            self.data = torch.stack([self.data, self.data, self.data], dim=1)

    def _prep_set(self):
        if self.train:
            tr_data, val_data = _train_val_split(*_get_sorted_data(self))
            tr_data = self._filter_labels(tr_data, self.keep_classes)
            val_data = self._filter_labels(val_data, self.keep_classes)
            if self.which_set == "train":
                self.data = torch.from_numpy(tr_data[0])
                self.targets = torch.from_numpy(tr_data[1])
            else:    # split == "valid"
                self.data = torch.from_numpy(val_data[0])
                self.targets = torch.from_numpy(val_data[1])
        else:
            d, t = _get_sorted_data(self)    # used for consistency -- transpose and sorting done in this fn.
            d, t = self._filter_labels((d, t), self.keep_classes)
            self.data = torch.from_numpy(d)
            self.targets = torch.from_numpy(t)

    @staticmethod
    def _filter_labels(data, keep_classes):
        d, t = data
        if keep_classes is not None:
            keep_inds = np.isin(t, np.array(keep_classes))
            d = d[keep_inds]
            t = t[keep_inds]
        return d, t

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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


class StaticEMNIST_Idx(StaticEMNIST):
    def __getitem__(self, index):  # Overwrite DatasetFolder getitem
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def _get_emnist_datasets(root_dir, as_datasets=False, transform=None, norm=False, color=False):
    print(root_dir)
    sorted_tr_data = _get_sorted_data(EMNIST(root_dir, "balanced", train=True))
    sorted_tst_data = _get_sorted_data(EMNIST(root_dir, "balanced", train=False))
    sorted_tr_data, sorted_val_data = _train_val_split(*sorted_tr_data)
    if not as_datasets:
        return [sorted_tr_data, sorted_val_data, sorted_tst_data]

    tr_ds = EMNIST(root_dir, "balanced", train=True, transform=transform)
    val_ds = EMNIST(root_dir, "balanced", train=True, transform=transform)
    tst_ds = EMNIST(root_dir, "balanced", train=False, transform=transform)

    print(tr_ds.data.shape)
    print(tr_ds.data.max())

    tr_ds = _replace_data_and_targets(tr_ds, sorted_tr_data[0], sorted_tr_data[1], norm, color)
    val_ds = _replace_data_and_targets(val_ds, sorted_val_data[0], sorted_val_data[1], norm, color)
    tst_ds = _replace_data_and_targets(tst_ds, sorted_tst_data[0], sorted_tst_data[1], norm, color)

    print(tr_ds.data.shape)
    print(tr_ds.data.max())

    return tr_ds, val_ds, tst_ds


def _replace_data_and_targets(ds, data, targets, norm, color):
    # if norm:
    #     data = data.astype(np.float32) * float(2./255.) - 1.     # (0, 255) --> (-1, 1)
    # if color:
    #     data = data.reshape((-1, data.shape[1], data.shape[2], 1))  # [-1, 28, 28] --> [-1, 1, 28, 28]
    #     data = np.concatenate([data, data, data], axis=3)  # [-1, 1, 28, 28] --> [-1, 3, 28, 28]
    ds.data = torch.from_numpy(data)
    ds.targets = torch.from_numpy(targets)
    return ds


def _create_multi_background_dataset(bg_imgs, sorted_imgs, sorted_labels, save_path):
    current_lbl = "Init"
    i = 0
    for img, lbl in zip(sorted_imgs, sorted_labels):
        if lbl != current_lbl:
            i = 0
            current_lbl = lbl
            mkdir_p(os.path.join(save_path, str(current_lbl)))
        bg_img = np.random.choice(bg_imgs)
        c_img = dt.colour_background(img, bg_img)
        imsave(os.path.join(save_path, str(current_lbl), str(i)), c_img)
        i += 1
        break


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
        # transf_img = transform_fn(img)                                          # float32 array
        # PIL.Image.fromarray(np.uint8(transf_img)).save(img_save_path)           # save as int8 image (grey or colour)
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
    # To check if it worked can count the files in each corruption directory and check they are the same == 131745
    # https://stackoverflow.com/questions/15216370/how-to-count-number-of-files-in-each-directory
    # PARAMS
    parser = argparse.ArgumentParser(description='Generate multiple EMNIST corruptions in parallel.')
    parser.add_argument('--corruption-ID', type=int, default=0, help="which corruption to generate")
    args = parser.parse_args()

    create_datasets = True
    # REPRODUCIBILITY
    seed = 1234

    # PATHS
    root_dir = "/om2/user/imason/compositions/datasets/"
    output_dir = os.path.join(root_dir, "EMNIST4/")


    if create_datasets:
        # LOAD EMNIST DATA
        all_data = _get_emnist_datasets(root_dir)
        dset_names = ['train', 'valid', 'test']

        # CREATE TARGET DATASETS: E.G. ../datasets/mnist/EMNIST/inverse/train/0/21.jpg
        np.random.seed(seed)   # diff transforms for train, val and test sets.
        for (sorted_imgs, sorted_labels), dset_name in zip(all_data, dset_names):
            # SETUP TARGET DATASETS FROM CORRUPTIONS AND COMPOSITIONS IN PKL FILE
            with open(os.path.join(output_dir, "corruption_names.pkl"), "rb") as f:
                all_corruptions = pickle.load(f)

            # EMNIST4
            c_names = all_corruptions[args.corruption_ID]
            corrs = [getattr(dt, c)() for c in c_names]
            c_name = '-'.join([corr.name for corr in corrs])
            corruption_names = [c_name]
            corruption_paths = [os.path.join(output_dir, c_name, dset_name)]
            corruption_fns = [corrs]

            # Create datasets
            _create_emnist_target_datasets(sorted_imgs, sorted_labels, corruption_fns, corruption_names,
                                           corruption_paths)




