import os
import os.path as osp
import argparse
import numpy as np
from imageio import imsave
import pickle

import dnnlib
import dnnlib.tflib as tflib

from util.utilgan import latent_anima, load_latents, file_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2 on Tensorflow"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dlatents', default=None, help='Saved dlatent vectors in single *.npy file or directory with such files')
parser.add_argument('--style_npy_file', default=None, help='Saved latent vector for hi res (style) features')
parser.add_argument('--out_dir', default='_out', help='Output directory')
parser.add_argument('--model', default='models/ffhq-1024-f.pkl', help='path to checkpoint file')
parser.add_argument('--size', default=None, help='Output resolution')
parser.add_argument('--scale_type', choices = ['pad','padside','centr','side','fit'], default='centr', help="pad (from center or topleft); centr/side = first scale then pad")
parser.add_argument('--trunc', type=float, default=1, help='Truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--digress', type=float, default=0, help='distortion technique by Aydao (strength of the effect)') 
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ops', default='cuda', help='custom op implementation (cuda or ref)')
# animation
parser.add_argument("--fstep", type=int, default=25, help="Number of frames for smooth interpolation")
parser.add_argument("--cubic", action='store_true', help="Use cubic splines for smoothing")
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]

def main():
    os.makedirs(a.out_dir, exist_ok=True)

    # setup generator
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.func_name = 'training.stylegan2_multi.G_main'
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type
    Gs_kwargs.impl = a.ops
    
    # load model with arguments
    sess = tflib.init_tf({'allow_soft_placement':True})
    pkl_name = osp.splitext(a.model)[0]
    with open(pkl_name + '.pkl', 'rb') as file:
        network = pickle.load(file, encoding='latin1')
    try: _, _, network = network
    except: pass
    for k in list(network.static_kwargs.keys()):
        Gs_kwargs[k] = network.static_kwargs[k]

    # reload custom network, if needed
    if '.pkl' in a.model.lower(): 
        print(' .. Gs from pkl ..', basename(a.model))
        Gs = network
    else: # reconstruct network
        print(' .. Gs custom ..', basename(a.model))
        Gs = tflib.Network('Gs', **Gs_kwargs)
        Gs.copy_vars_from(network)

    z_dim = Gs.input_shape[1]
    dz_dim = 512 # dlatent_size
    try: dl_dim = 2 * (int(np.floor(np.log2(Gs_kwargs.resolution))) - 1)
    except: print(' Resave model, no resolution kwarg found!'); exit(1)
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
        # try replacing 5 by other value, less than dl_dim
        key_dlatents[:, :, range(5,dl_dim), :] = style_dlatent[:, :, range(5,dl_dim), :]
       
    frames = key_dlatents.shape[0] * a.fstep
    
    dlatents = latent_anima(dlat_shape, frames, a.fstep, key_latents=key_dlatents, cubic=a.cubic, verbose=True) # [frm,1,512]
    print(' dlatents', dlatents.shape)
    frame_count = dlatents.shape[0]

    # truncation trick
    dlatent_avg = Gs.get_var('dlatent_avg') # (512,)
    tr_range = range(0,8)
    dlatents[:,:,tr_range,:] = dlatent_avg + (dlatents[:,:,tr_range,:] - dlatent_avg) * a.trunc
    
    # distort image by tweaking initial const layer
    if a.digress > 0:
        try: latent_size = Gs.static_kwargs['latent_size']
        except: latent_size = 512 # default latent size
        try: init_res = Gs.static_kwargs['init_res']
        except: init_res = (4,4) # default initial layer size 
        dconst = a.digress * latent_anima([1, latent_size, *init_res], frames, a.fstep, cubic=True, verbose=False)
    else:
        dconst = np.zeros([frame_count, 1, 1, 1, 1])

    # generate images from latent timeline
    pbar = ProgressBar(frame_count)
    for i in range(frame_count):
    
        if a.digress is True:
            tf.get_default_session().run(tf.assign(wvars[0], wts[i]))

        # generate multi-latent result
        if Gs.num_inputs == 2:
            output = Gs.components.synthesis.run(dlatents[i], randomize_noise=False, output_transform=fmt, minibatch_size=1)
        else:
            output = Gs.components.synthesis.run(dlatents[i], [None], dconst[i], randomize_noise=False, output_transform=fmt, minibatch_size=1)

        ext = 'png' if output.shape[3]==4 else 'jpg'
        filename = osp.join(a.out_dir, "%06d.%s" % (i,ext))
        imsave(filename, output[0])
        pbar.upd()

        
if __name__ == '__main__':
    main()

