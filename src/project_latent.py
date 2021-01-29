# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
# This work is made available under the Nvidia Source Code License-NC.
# https://nvlabs.github.io/stylegan2/license.html

import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import cv2, PIL
import pickle

import projector
from training import misc

from util.utilgan import img_list, img_read, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', help='Network pickle filename', required=True)
parser.add_argument('--in_dir', help='Dataset root directory', required=True)
parser.add_argument('--out_dir', help='Root directory for run results (default: %(default)s)', default='_out', metavar='DIR')
parser.add_argument('--steps', type=int, default=1000, help='Number of iterations (default: %(default)s)') # 1000
parser.add_argument('--num_snapshots', type=int, default=10, help='Number of snapshots (default: %(default)s)')
a = parser.parse_args()

def write_video_frame(proj, video_out):
    img = proj.get_images()[0]
    img = misc.convert_to_pil_image(img, drange=[-1, 1])
    video_frame = img # .resize((512, 512))
    video_out.write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))

def project_image(proj, targets, work_dir, resolution, num_snapshots):
    filename = osp.join(work_dir, basename(work_dir))
    video_out = cv2.VideoWriter(filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, resolution)
    
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, filename + '.jpg', drange=[-1,1])
    proj.start(targets)
    pbar = ProgressBar(proj.num_steps)
    while proj.get_cur_step() < proj.num_steps:
        proj.step()
        write_video_frame(proj, video_out)
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), filename + '-%04d.jpg' % proj.get_cur_step(), drange=[-1,1])
        pbar.upd()

    dlats = proj.get_dlatents()
    np.save(filename + '-%04d.npy' % proj.get_cur_step(), dlats)
    video_out.release()

def main():
    print('Loading networks from "%s"...' % a.model)
    sess = tflib.init_tf()
    with open(a.model, 'rb') as file:
        network = pickle.load(file, encoding='latin1')
        try: _, _, Gs = network
        except:    Gs = network
    resolution = tuple(Gs.output_shape[2:])
    proj = projector.Projector(a.steps)
    proj.set_network(Gs)

    img_files = img_list(a.in_dir)
    num_images = len(img_files)
    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx+1, num_images))
        images = img_read(img_files[image_idx])
        images = np.expand_dims(np.transpose(images, [2,0,1]), 0)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        work_dir = osp.join(a.out_dir, basename(img_files[image_idx]))
        os.makedirs(work_dir, exist_ok=True)
        project_image(proj, images, work_dir, resolution, a.num_snapshots)
    
    
if __name__ == "__main__":
    main()
