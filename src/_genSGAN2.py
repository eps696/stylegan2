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

from util.utilgan import latent_anima, basename
from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2 on Tensorflow"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--out_dir', default='_out', help='Output directory')
parser.add_argument('--model', default='models/ffhq-1024-f.pkl', help='path to checkpoint file')
parser.add_argument('--size', default=None, help='Output resolution')
parser.add_argument('--scale_type', choices = ['centr','side','fit'], default='centr', help="fit or pad (from centr or left)")
parser.add_argument('--trunc', type=float, default=0.8, help='Truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--latent_size', type=int, default=512)
parser.add_argument('--ops', default='cuda', help='Custom op implementation (cuda or ref)')
# animation
parser.add_argument('--frames', default='200-25', help='how many frames to generate')
parser.add_argument("--cubic", action='store_true', help="Use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="Use Gaussian smoothing")
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

def main():
    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(seed=696)

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
    Gs_kwargs.impl = a.ops
    
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
    # Gs.print_layers()
    print(' out shape', Gs.output_shape[1:])

    z_dim = Gs.input_shape[1]
    shape = (1, z_dim)
    
    print(' making timeline..')
    latents = latent_anima(shape, a.frames, a.fstep, cubic=a.cubic, gauss=a.gauss, verbose=True) # [frm,1,512]
    print(' latents', latents.shape)
    
    # generate images from latent timeline
    frame_count = latents.shape[0]
    pbar = ProgressBar(frame_count)
    for i in range(frame_count):
        output = Gs.run(latents[i], [None], truncation_psi=a.trunc, randomize_noise=False, output_transform=fmt)
        ext = 'png' if output.shape[3]==4 else 'jpg'
        filename = osp.join(a.out_dir, "%05d.%s" % (i,ext))
        imsave(filename, output[0])
        pbar.upd()

    # convert latents to dlatents, save them
    latents = latents.squeeze(1) # [frm,512]
    dlatents = Gs.components.mapping.run(latents, None, latent_size=z_dim, dtype='float16') # [frm,18,512]
    filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
    filename = osp.join(osp.dirname(a.out_dir), filename)
    np.save(filename, dlatents)
    print('saved dlatents', dlatents.shape, 'to', filename)
        
if __name__ == '__main__':
    main()
