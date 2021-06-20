import os
import warnings
warnings.filterwarnings("ignore")
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np
import cv2

from utilgan import img_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_dir', help='Input directory')
parser.add_argument('-o', '--out_dir', help='Output directory')
parser.add_argument('-s', '--size', type=int, default=512, help='Output size in pixels')
parser.add_argument('--step', type=int, default=None, help='Step to shift between crops')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--png_compression', type=int, default=1, help='png compression (0 to 9; 0 = uncompressed, fast)')
parser.add_argument('--jpg_quality', type=int, default=95, help='jpeg quality (0 to 100; 95 = max reasonable)')
parser.add_argument('-d', '--down', action='store_true', help='Downscale before crop? (smaller side to size)')
parser.add_argument('--ext', default=None, help='Override output format')
a = parser.parse_args()

# https://pillow.readthedocs.io/en/3.0.x/handbook/image-file-formats.html#jpeg
# image quality = from 1 (worst) to 95 (best); default 75. Values above 95 should be avoided; 
# 100 disables portions of the JPEG compression algorithm => results in large files with hardly any gain in image quality.

# CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
# compression time. If read raw images during training, use 0 for faster IO speed.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def crop_step(path, out_dir, out_size, step, min_step):
    img_name = basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # convert monochrome to RGB if needed
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    if img.shape[2] == 1:
        img = img[:, :, (0,0,0)]
    h, w, c = img.shape
    
    ext = a.ext if a.ext is not None else 'png' if img.shape[2]==4 else 'jpg'
    compr = [cv2.IMWRITE_PNG_COMPRESSION, a.png_compression] if ext=='png' else [cv2.IMWRITE_JPEG_QUALITY, a.jpg_quality]

    min_size = min(h,w)
    if min_size < out_size:
        h = int(h * out_size/min_size)
        w = int(w * out_size/min_size)
        img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
    elif min_size > out_size and a.down is True:
        h = int(h * out_size/min_size)
        w = int(w * out_size/min_size)
        img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
        
    h_space = np.arange(0, h - out_size + 1, step)
    if h - (h_space[-1] + out_size) < min_step:
        h_space = h_space[:-1]
    h_space = np.append(h_space, h - out_size)
    w_space = np.arange(0, w - out_size + 1, step)
    if w - (w_space[-1] + out_size) < min_step:
        w_space = w_space[:-1]
    w_space = np.append(w_space, w - out_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_img = img[x:x + out_size, y:y + out_size, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(os.path.join(out_dir, '%s-s%03d.%s' % (img_name, index, ext)), crop_img, compr)
    return 'Processing {:s} ...'.format(img_name)

def main():
    """A multi-thread tool to crop sub images."""
    os.makedirs(a.out_dir, exist_ok=True)
    images = img_list(a.in_dir, subdir=True)

    step = a.size // 2 if a.step is None else a.step
    min_step = step // 4

    def update(arg):
        pbar.upd(arg)

    pbar = ProgressBar(len(images))
    pool = Pool(a.workers)
    for path in images:
        pool.apply_async(crop_step, args=(path, a.out_dir, a.size, step, min_step), callback=update)
    pool.close()
    pool.join()
    print('All done')


if __name__ == '__main__':
    # workaround for multithreading in jupyter console
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
