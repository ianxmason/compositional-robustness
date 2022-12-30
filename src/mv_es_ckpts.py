"""
Quick script to rename all the es ckpts for testing
This is just for initial check that everything is working. In reality the ES ckpts should get deleted when fully
trained.
"""

import os

ckpt_dir = '/om2/user/imason/compositions/ckpts/FACESCRUB'

for f in os.listdir(ckpt_dir):
    if 'es_ckpt' in f:
        new_name = f.replace('es_ckpt_', '')
        new_name = new_name.replace('.pt.pt', '.pt')
        os.rename(os.path.join(ckpt_dir, f), os.path.join(ckpt_dir, new_name))
