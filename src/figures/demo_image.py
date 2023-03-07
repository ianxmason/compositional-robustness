"""
Hardcode some image corruptions for a slide to demonstrate the compositional nature of visual understanding.
"""
import os
import sys
import PIL.Image
sys.path.append("../")
import data.data_transforms as dt
import numpy as np

img_path = "/Users/ian/Presentations/Fujitsu/Composed_Corruptions_Images"
img_name = "Golden_Retriever_Draw"


# Use quantize, saturate, contrast with severity 3
c_names = ['impulse_noise', 'gaussian_blur', 'quantize', 'rotate', 'scale', 'contrast', 'saturate']
transform_fns = [getattr(dt, c) for c in c_names]

img = PIL.Image.open(os.path.join(img_path, img_name + ".png"))

img_transforms = ""
for i, transform_fn in enumerate(transform_fns):
    img_transforms = "_".join(c_names[:i+1])
    img_save_path = os.path.join(img_path, img_name + "_" + img_transforms + ".png")
    img = transform_fn(img)
    PIL.Image.fromarray(np.uint8(img)).save(img_save_path)
    print("Transformed image saved to:", img_save_path)