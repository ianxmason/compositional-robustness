"""
Gets a few of the elemental corruption images to use in figures
Use these images: CIFAR: 2.png or 3.png. EMNIST: 14.png or 3.png FACESCRUB 20.png or 76.png
"""
import sys
import PIL.Image
import numpy as np
sys.path.append("../")
import data.data_transforms as dt
from lib.utils import mkdir_p

dset = "FACESCRUB"
base_img_path = f"/om2/user/imason/compositions/datasets/{dset}/Identity/train/20/76.png"
save_path = f"/om2/user/imason/compositions/figs/{dset}/example_data/"
mkdir_p(save_path)

elemental_corruptions = ['Contrast', 'GaussianBlur', 'ImpulseNoise', 'Invert', 'Rotate90', 'Swirl', 'Identity']
example_compositions = [['Contrast', 'GaussianBlur'], ['ImpulseNoise', 'Invert'], ['Rotate90', 'Swirl'],
                        ['Rotate90', 'ImpulseNoise', 'Contrast'], ['Swirl', 'Invert', 'GaussianBlur'],
                        ['Invert', 'Rotate90', 'Contrast', 'Swirl'],
                        ['GaussianBlur', 'Rotate90', 'Swirl', 'Contrast'],
                        ['Contrast', 'GaussianBlur', 'ImpulseNoise', 'Invert', 'Rotate90', 'Swirl']]

for corruption_name in elemental_corruptions:
    transforms = [getattr(dt, corruption_name)()]
    base_img = PIL.Image.open(base_img_path)
    for t in transforms:
        base_img = t(base_img)
    PIL.Image.fromarray(np.uint8(base_img)).save(save_path + corruption_name + ".png")

for corruption_list in example_compositions:
    corruption_name = "-".join(corruption_list)
    transforms = [getattr(dt, c)() for c in corruption_list]
    base_img = PIL.Image.open(base_img_path)
    for t in transforms:
        base_img = t(base_img)
    PIL.Image.fromarray(np.uint8(base_img)).save(save_path + corruption_name + ".png")