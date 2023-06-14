"""
Data transforms.
Some are adapted from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py
"""
from abc import ABC, abstractmethod
import math
from scipy.ndimage.interpolation import shift
from scipy.ndimage import grey_erosion, grey_dilation, gaussian_filter, gaussian_filter1d
import scipy.ndimage
import torch
import numpy as np
import warnings
from matplotlib.image import imread
import skimage as sk
from skimage.filters import gaussian
from skimage import transform, feature
from PIL import Image as PILImage
import cv2
import warnings
import os

class Corruption(ABC):
    """
    Abstract class for corruption transforms.

    Needs attributes: name, abbreviation, lossless

    Lossless is in pixel space (e.g. scale is lossless in continuous space but not in discrete pixel space)
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def abbreviation(self):
        pass

    @property
    @abstractmethod
    def lossless(self):
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class Identity(Corruption):
    def __init__(self):
        pass

    @property
    def name(self):
        return 'Identity'

    @property
    def abbreviation(self):
        return 'ID'

    @property
    def lossless(self):
        return True

    def __call__(self, x):
        return np.array(x).astype(np.float32)


class GaussianBlur(Corruption):
    def __init__(self, severity=2):
        if severity not in [1, 2, 3, 4, 5]:
            raise ValueError("Severity must be between 1 and 5.")
        self.severity = severity

    @property
    def name(self):
        return 'GaussianBlur'

    @property
    def abbreviation(self):
        return 'GB'

    @property
    def lossless(self):
        return False

    def __call__(self, x):
        c = [1, 2, 3, 4, 6][self.severity - 1]
        # At time of creation easiest is to let the function guess channel_axis
        # some datasets have images (N,N) and others are (N,N,3). The current implementation allows for this
        # https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian
        # https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/_shared/filters.py#L16-L137
        # But this does raise a warning which we suppress.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = gaussian(np.array(x) / 255., sigma=c)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)


class ImpulseNoise(Corruption):
    def __init__(self, severity=5):
        if severity not in [1, 2, 3, 4, 5]:
            raise ValueError("Severity must be between 1 and 5.")
        self.severity = severity

    @property
    def name(self):
        return 'ImpulseNoise'

    @property
    def abbreviation(self):
        return 'IM'

    @property
    def lossless(self):
        return False

    def __call__(self, x):
        c = [.03, .06, .09, 0.17, 0.27][self.severity - 1]
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)


class Invert(Corruption):
    def __init__(self):
        pass

    @property
    def name(self):
        return 'Invert'

    @property
    def abbreviation(self):
        return 'IN'

    @property
    def lossless(self):
        return True

    def __call__(self, x):
        x = np.array(x).astype(np.float32)
        return 255. - x


class Rotate90(Corruption):
    def __init__(self):
        pass

    @property
    def name(self):
        return 'Rotate90'

    @property
    def abbreviation(self):
        return 'R90'

    @property
    def lossless(self):
        return True

    def __call__(self, x):
        x = np.array(x) / 255.
        angle = math.pi / 2.
        center = ((x.shape[0] / 2) - 0.5, (x.shape[1] / 2) - 0.5)

        aff = transform.AffineTransform(rotation=angle)

        a1, a2 = aff.params[0, :2]
        b1, b2 = aff.params[1, :2]
        a3 = center[0] * (1 - a1 - a2)
        b3 = center[1] * (1 - b1 - b2)
        aff = transform.AffineTransform(rotation=angle, translation=[a3, b3])

        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)


class Contrast(Corruption):
    def __init__(self, severity=2):
        if severity not in [1, 2, 3, 4, 5]:
            raise ValueError("Severity must be between 1 and 5.")
        self.severity = severity

    @property
    def name(self):
        return 'Contrast'

    @property
    def abbreviation(self):
        return 'CO'

    @property
    def lossless(self):
        return False

    def __call__(self, x):
        c = [0.4, .3, .2, .1, .05][self.severity - 1]
        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        x = np.clip((x - means) * c + means, 0, 1) * 255
        return x.astype(np.float32)


class Swirl(Corruption):
    def __init__(self, severity=3):
        if severity not in [1, 2, 3, 4, 5]:
            raise ValueError("Severity must be between 1 and 5.")
        self.severity = severity

    @property
    def name(self):
        return 'Swirl'

    @property
    def abbreviation(self):
        return 'SW'

    @property
    def lossless(self):
        return False  # Not certain

    def __call__(self, x):
        c = [0.5, 1, 3, 5, 10][self.severity - 1]
        x = np.array(x) / 255.
        x = transform.swirl(x, rotation=0, strength=c, radius=math.sqrt(2) * x.shape[0] / 2)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)


# ---------------------------------- GENERAL -----------------------------------

def denormalize_255(x, mean, std):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    # After denormalization the values are in range [0, 1]. To visualize them we want [0, 255].
    return x * 255.
