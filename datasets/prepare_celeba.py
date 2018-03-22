from glob import glob
import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.misc import imread, imresize

filenames = glob(os.path.join("img_align_celeba", "*.jpg"))
filenames = np.sort(filenames)
w, h = 64, 64  # Change this if you wish to use larger images
data = np.zeros((len(filenames), w * h * 3), dtype=np.uint8)

# This preprocessing is appriate for CelebA but should be adapted
# (or removed entirely) for other datasets.


def get_image(image_path, w=64, h=64):
    im = imread(image_path).astype(np.float)
    orig_h, orig_w = im.shape[:2]
    new_h = int(orig_h * w / orig_w)
    im = imresize(im, (new_h, w))
    margin = int(round((new_h - h)/2))
    return im[margin:margin+h]

for n, fname in tqdm(enumerate(filenames)):
    image = get_image(fname)
    data[n] = image.flatten()

with h5py.File('datasets/celeba.h5', 'w') as f:
    f.create_dataset("images", data=data)
