# Original copyright (c) 2019, NVIDIA Corporation. All rights reserved.
# This work is made available under the Nvidia Source Code License-NC
# https://nvlabs.github.io/stylegan2/license.html

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import argparse
import copy
import numpy as np
import tensorflow as tf

import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from training import dataset as _dataset
from training.dataset_tool import create_from_images

from util.utilgan import basename, file_list

def run(dataset, train_dir, config, d_aug, diffaug_policy, cond, ops, jpg_data, mirror, mirror_v, \
        lod_step_kimg, batch_size, resume, resume_kimg, finetune, num_gpus, ema_kimg, gamma, freezeD):

    # training functions
    if d_aug: # https://github.com/mit-han-lab/data-efficient-gans
        train = EasyDict(run_func_name='training.training_loop_diffaug.training_loop')          # Options for training loop (Diff Augment method)
        loss_args = EasyDict(func_name='training.loss_diffaug.ns_DiffAugment_r1', policy=diffaug_policy) # Options for loss (Diff Augment method)
    else: # original nvidia
        train = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop (original from NVidia)
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')     # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1')             # Options for discriminator loss.
    
    # network functions
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                        # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().
    G.impl    = D.impl = ops

    # dataset (tfrecords) - get or create
    tfr_files = file_list(os.path.dirname(dataset), 'tfr')
    tfr_files = [f for f in tfr_files if basename(f).split('-')[0] == basename(dataset)]
    if len(tfr_files) == 0 or os.stat(tfr_files[0]).st_size == 0:
        tfr_file, total_samples = create_from_images(dataset, jpg=jpg_data)
    else:
        tfr_file = tfr_files[0]
    dataset_args = EasyDict(tfrecord=tfr_file, jpg_data=jpg_data)
    
    # resolutions
    with tf.Graph().as_default(), tflib.create_session().as_default(): # pylint: disable=not-context-manager
        dataset_obj = _dataset.load_dataset(**dataset_args) # loading the data to see what comes out
        resolution = dataset_obj.resolution
        init_res = dataset_obj.init_res
        res_log2 = dataset_obj.res_log2
        dataset_obj.close()
        dataset_obj = None
    
    if list(init_res) == [4,4]: 
        desc = '%s-%d' % (basename(dataset), resolution)
    else:
        print(' custom init resolution', init_res)
        desc = basename(tfr_file)
    G.init_res = D.init_res = list(init_res)
    
    train.savenames = [desc.replace(basename(dataset), 'snapshot'), desc]
    desc += '-%s' % config
    
    # training schedule
    sched.lod_training_kimg = lod_step_kimg 
    sched.lod_transition_kimg = lod_step_kimg 
    sched.tick_kimg_base = 2 # if finetune else 0.2
    train.total_kimg = lod_step_kimg * res_log2 * 3 # 1.5 * ProGAN
    train.image_snapshot_ticks = 1
    train.network_snapshot_ticks = 5
    train.mirror_augment = mirror
    train.mirror_augment_v = mirror_v

    # learning rate 
    if config == 'e':
        if finetune: # uptrain 1024
            sched.G_lrate_base = 0.001
            sched.lrate_step = 150 # period for stepping to next lrate, in kimg
        else: # train 1024
            sched.G_lrate_base = 0.001
            sched.G_lrate_dict = {0:0.001, 1:0.0007, 2:0.0005, 3:0.0003}
            sched.lrate_step = 1500 # period for stepping to next lrate, in kimg
    if config == 'f':
        sched.G_lrate_base = 0.001 # 0.0003 for few-shot datasets
    sched.D_lrate_base = sched.G_lrate_base # *2 - not used anyway

    sched.minibatch_gpu_base = batch_size
    sched.minibatch_size_base = num_gpus * sched.minibatch_gpu_base
    sc.num_gpus = num_gpus
    
    if config == 'e':
        G.fmap_base = D.fmap_base = 8 << 10
        if d_aug: loss_args.gamma = 100 if gamma is None else gamma
        else: D_loss.gamma = 100 if gamma is None else gamma
    elif config == 'f':
        G.fmap_base = D.fmap_base = 16 << 10
    else:
        print(' Only configs E and F are implemented'); exit()

    if cond:
        desc += '-cond'; dataset_args.max_label_size = 'full' # conditioned on full label

    if freezeD: 
        D.freezeD = True
        train.resume_with_new_nets = True

    if d_aug: 
        desc += '-daug'

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, tf_config=tf_config)
    kwargs.update(resume_pkl=resume, resume_kimg=resume_kimg, resume_with_new_nets=True)
    if ema_kimg is not None:
        kwargs.update(G_ema_kimg=ema_kimg)
    if d_aug: 
        kwargs.update(loss_args=loss_args)
    else:
        kwargs.update(G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = train_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='StyleGAN2 with tweaks', formatter_class=argparse.RawDescriptionHelpFormatter)
    # main
    parser.add_argument('--dataset', required=True, help='Training dataset path', metavar='DIR')
    parser.add_argument('--train_dir', default='train', help='Root directory for training results (default: %(default)s)', metavar='DIR')
    parser.add_argument('--resume', default=None, help='Resume checkpoint path. None = from scratch')
    parser.add_argument('--resume_kimg', default=0, type=int, help='Resume training from (in thousands of images)', metavar='N')
    parser.add_argument('--lod_step_kimg', default=20, type=int, help='Thousands of images per LOD/layer step (default: %(default)s)', metavar='N')
    parser.add_argument('--finetune', action='store_true', help='finetune trained model (start from 1e4 kimg, stop when enough)')
    # network
    parser.add_argument('--config', default='F', help='Training config E (shrink) or F (large) (default: %(default)s)', metavar='CONFIG')
    parser.add_argument('--ops', default='cuda', help='Custom op implementation (cuda or ref, default: %(default)s)')
    parser.add_argument('--gamma', default=None, type=float, help='R1 regularization weight')
    # special
    parser.add_argument('--jpg_data', action='store_true', help='Dataset tfrecords in raw JPG?')
    parser.add_argument('--d_aug', action='store_true', help='Use Diff Augment training for small datasets')
    parser.add_argument('--diffaug_policy', default='color,translation,cutout', help='Comma-separated list of DiffAugment policies (default: %(default)s)', metavar='..')
    parser.add_argument('--ema_kimg', default=None, type=int, help='Half-life of exponential moving average (for Diff Augment)', metavar='N')
    parser.add_argument('--freezeD', action='store_true', help='freeze lower D layers for better finetuning')
    parser.add_argument('--cond', action='store_true', help='conditional model')
    # training
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU (default: %(default)s)', metavar='N')
    parser.add_argument('--mirror', help='Mirror augment (default: %(default)s)', default=True, metavar='BOOL', type=bool)
    parser.add_argument('--mirror_v', help='Mirror augment vertically (default: %(default)s)', default=False, metavar='BOOL', type=bool)
    parser.add_argument('--num_gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    args = parser.parse_args()

    args.config = args.config.lower()

    run(**vars(args))


if __name__ == "__main__":
    main()
