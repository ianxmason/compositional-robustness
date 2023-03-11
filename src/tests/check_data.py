"""
Takes one file from each corruption/composition and each of train/val/test and create a much smaller dataset that
can be copied quickly to a local machine to visualise/play with
"""
import os
import sys
import shutil
sys.path.append("../")
from lib.utils import mkdir_p

data_dir = '/om2/user/imason/compositions/datasets/EMNIST/'
save_dir = '/om2/user/imason/compositions/check_data/EMNIST/'
mkdir_p(save_dir)

for corruptions in os.listdir(data_dir):
    if corruptions == "raw":
        continue
    if os.path.isdir(os.path.join(data_dir, corruptions)):
        mkdir_p(os.path.join(save_dir, corruptions))
        for split in ['train', 'valid', 'test']:
            mkdir_p(os.path.join(save_dir, corruptions, split))
            img_path = os.path.join(data_dir, corruptions, split, '7', '0.png')
            # Copy image to save_dir
            shutil.copyfile(img_path, os.path.join(save_dir, corruptions, split, '0.png'))
