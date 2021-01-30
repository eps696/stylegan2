import os
import sys
from multiprocessing import Pool
from shutil import get_terminal_size
import time
import argparse

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
parser.add_argument('-s', '--size', type=int, default=512, help='Output directory')
parser.add_argument('--step', type=int, default=None, help='Step')
parser.add_argument('--workers', type=int, default=8, help='number of workers (8, as of cpu#)')
a = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def worker(path, save_folder, crop_size, step, min_step, compression_level):
    img_name = basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # convert monochrome to RGB if needed
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    if img.shape[2] == 1:
        img = img[:, :, (0,0,0)]
    h, w, c = img.shape
    
    ext = 'png' if img.shape[2]==4 else 'jpg'

    min_size = min(h,w)
    if min_size < crop_size:
        h = int(h * crop_size/min_size)
        w = int(w * crop_size/min_size)
        img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
        
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > min_step:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > min_step:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_img = img[x:x + crop_size, y:y + crop_size, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(os.path.join(save_folder, '%s-s%03d.%s' % (img_name, index, ext)), crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)

def main():
    """A multi-thread tool to crop sub images."""
    input_folder = a.in_dir
    save_folder = a.out_dir
    n_thread = a.workers
    crop_size = a.size
    step = a.size // 2 if a.step is None else a.step
    min_step = a.size // 8
    compression_level = 1  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    os.makedirs(save_folder, exist_ok=True)

    images = img_list(input_folder, subdir=True)

    def update(arg):
        pbar.upd(arg)

    pbar = ProgressBar(len(images))

    pool = Pool(n_thread)
    for path in images:
        pool.apply_async(worker,
            args=(path, save_folder, crop_size, step, min_step, compression_level),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    # workaround for multithreading in jupyter console
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()

