import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import numpy as np
from imageio import imsave
import pickle

import tensorflow as tf
gpu=tf.test.gpu_device_name(); print('.. TF GPU %s' % tf.__version__ if gpu is not None else '!.GPU not found.!')

import dnnlib
import dnnlib.tflib as tflib

from util.utilgan import latent_anima, load_latents, file_list, basename
from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2 on Tensorflow"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dlatents', default=None, help='Saved dlatent vectors in single *.npy file or directory with such files')
parser.add_argument('--style_npy_file', default=None, help='Saved latent vector for base features')
parser.add_argument('--out_dir', default='_out', help='Output directory')
parser.add_argument('--model', default='models/ffhq-1024-f.pkl', help='path to checkpoint file')
parser.add_argument('--size', default=None, help='Output resolution')
parser.add_argument('--scale_type', choices = ['fit','centr','side'], default='centr', help="fit or pad (centr or from left)")
parser.add_argument('--trunc', type=float, default=1, help='Truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--latent_size', type=int, default=512)
parser.add_argument('--dlatent_size', type=int, default=512)
# animation
parser.add_argument("--fstep", type=int, default=25, help="Number of frames for smooth interpolation")
parser.add_argument("--cubic", action='store_true', help="Use cubic splines for smoothing")
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]

def main():
    os.makedirs(a.out_dir, exist_ok=True)

    # parse filename to model parameters
    mparams = basename(a.model).split('-')
    res = int(mparams[1])
    cfg = mparams[2]
    
    # setup generator
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.func_name = 'training.stylegan2_custom.G_main'
    Gs_kwargs.verbose = False
    Gs_kwargs.resolution = res
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type
    Gs_kwargs.latent_size = a.latent_size
    
    if cfg.lower() == 'f':
        Gs_kwargs.synthesis_func = 'G_synthesis_stylegan2'
    elif cfg.lower() == 'e':
        Gs_kwargs.synthesis_func = 'G_synthesis_stylegan2'
        Gs_kwargs.fmap_base = 8 << 10
    else:
        print(' old modes [A-D] not implemented'); exit()
    
    # check initial model resolution
    if len(mparams) > 3: 
        if 'x' in mparams[3].lower():
            init_res = [int(x) for x in mparams[3].lower().split('x')]
            Gs_kwargs.init_res = list(reversed(init_res)) # [H,W]

    # load model, check channels
    sess = tflib.init_tf({'allow_soft_placement':True})
    pkl_name = osp.splitext(a.model)[0]
    with open(pkl_name + '.pkl', 'rb') as file:
        network = pickle.load(file, encoding='latin1')
    try: _, _, Gs = network
    except:    Gs = network
    Gs_kwargs.num_channels = Gs.output_shape[1]

    # reload custom network, if needed
    if '.pkl' in a.model.lower(): 
        print(' .. Gs from pkl ..')
    else: 
        print(' .. Gs custom ..')
        Gs = tflib.Network('Gs', **Gs_kwargs)
        Gs.copy_vars_from(network)

    z_dim = Gs.input_shape[1]
    dz_dim = a.dlatent_size # 512
    dl_dim = 2 * (int(np.floor(np.log2(res))) - 1)
    dlat_shape = (1, dl_dim, dz_dim) # [1,18,512]
    
    # read saved latents
    if a.dlatents is not None and osp.isfile(a.dlatents):
        key_dlatents = load_latents(a.dlatents)
        if len(key_dlatents.shape) == 2: key_dlatents = np.expand_dims(key_dlatents, 0)
    elif a.dlatents is not None and osp.isdir(a.dlatents):
        # if a.dlatents.endswith('/') or a.dlatents.endswith('\\'): a.dlatents = a.dlatents[:-1]
        key_dlatents = []
        npy_list = file_list(a.dlatents, 'npy')
        for npy in npy_list: 
            key_dlatent = load_latents(npy)
            if len(key_dlatent.shape) == 2: key_dlatent = np.expand_dims(key_dlatent, 0)
            key_dlatents.append(key_dlatent)
        key_dlatents = np.concatenate(key_dlatents) # [frm,18,512]
    else:
        print(' No input dlatents found'); exit()
    key_dlatents = key_dlatents[:, np.newaxis] # [frm,1,18,512]
    print(' key dlatents', key_dlatents.shape)
    
    # replace higher layers with single (style) latent
    if a.style_npy_file is not None:
        print(' styling with latent', a.style_npy_file)
        style_dlatent = load_latents(a.style_npy_file)
        while len(style_dlatent.shape) < 4: style_dlatent = np.expand_dims(style_dlatent, 0)
        # try other values < dl_dim besides 5
        key_dlatents[:, :, range(5,dl_dim), :] = style_dlatent[:, :, range(5,dl_dim), :]
       
    frames = key_dlatents.shape[0] * a.fstep
    
    dlatents = latent_anima(dlat_shape, frames, a.fstep, key_latents=key_dlatents, cubic=a.cubic, verbose=True) # [frm,1,512]
    print(' dlatents', dlatents.shape)

    # truncation trick
    dlatent_avg = Gs.get_var('dlatent_avg') # (512,)
    tr_range = range(0,8)
    dlatents[:,:,tr_range,:] = dlatent_avg + (dlatents[:,:,tr_range,:] - dlatent_avg) * a.trunc
    
    # loop for graph frame by frame
    frame_count = dlatents.shape[0]
    pbar = ProgressBar(frame_count)
    for i in range(frame_count):
    
        dlatent = dlatents[i]

        output = Gs.components.synthesis.run(dlatent, randomize_noise=False, output_transform=fmt, minibatch_size=1)

        ext = 'png' if output.shape[3]==4 else 'jpg'
        filename = osp.join(a.out_dir, "%05d.%s" % (i,ext))
        imsave(filename, output[0])
        pbar.upd()

        
if __name__ == '__main__':
    main()

