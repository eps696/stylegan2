import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings("ignore")
import argparse
import pickle
from collections import OrderedDict
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil

import tensorflow; tf = tensorflow.compat.v1 if hasattr(tensorflow.compat, 'v1') else tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

from util.utilgan import basename, calc_init_res
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Source model path')
parser.add_argument('--out_dir', default='./', help='Output directory for reduced/reconstructed model')
parser.add_argument('-r', '--reconstruct', action='store_true', help='Reconstruct model (add internal arguments)')
parser.add_argument('-s', '--res', default=None, help='Target resolution in format X-Y')
parser.add_argument('-a', '--alpha', action='store_true', help='Add alpha channel for RGBA processing')
parser.add_argument('-l', '--labels', default=None, type=int, help='Labels for conditional model')
parser.add_argument('-f', '--full', action='store_true', help='Save full model')
parser.add_argument('-v', '--verbose', action='store_true')
a = parser.parse_args()

if a.res is not None: 
    a.res = [int(s) for s in a.res.split('-')][::-1]
    if len(a.res) == 1: a.res = a.res + a.res

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        networks = pickle.load(f, encoding='latin1')
    try:
        G, D, Gs = networks
    except:
        Gs = networks
        G = D = None
    return G, D, Gs
    
def save_pkl(networks, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(networks, file, protocol=pickle.HIGHEST_PROTOCOL)

def create_model(data_shape, full=False, labels=None, kwargs_in=None):
    init_res, resolution, res_log2 = calc_init_res(data_shape[1:])
    kwargs_out = dnnlib.EasyDict()
    kwargs_out.num_channels = data_shape[0]
    if kwargs_in is not None:
        for k in list(kwargs_in.keys()):
            kwargs_out[k] = kwargs_in[k]
    if labels is not None: kwargs_out.label_size = labels
    kwargs_out.resolution = resolution
    kwargs_out.init_res = init_res
    if a.verbose is True: print(['%s: %s'%(kv[0],kv[1]) for kv in sorted(kwargs_out.items())])
    if full is True:
        G = tflib.Network('G', func_name='training.networks_stylegan2.G_main', **kwargs_out)
        D = tflib.Network('D', func_name='training.networks_stylegan2.D_stylegan2', **kwargs_out)
        Gs = G.clone('Gs')
    else:
        Gs = tflib.Network('Gs', func_name='training.networks_stylegan2.G_main', **kwargs_out)
        G = D = None
    return G, D, Gs

def copy_vars(src_net, tgt_net, D=False):
    names = [name for name in tgt_net.trainables.keys() if name in src_net.trainables.keys()]
    var_dict = OrderedDict()

    for name in names:
        if tgt_net.vars[name].shape == src_net.vars[name].shape: # fixing rgb-to-rgba only !!
            var_dict[name] = src_net.vars[name]
        else:
            var_dict[name] = add_channel(src_net.vars[name], D=D)

    weights_to_copy = {tgt_net.vars[name]: var_dict[name] for name in names}
    tfutil.set_vars(tfutil.run(weights_to_copy))

def add_channel(x, D=False): # [BCHW]
    if D is True: # pad dim before last [-2]
        padding = [[0,1],[0,0]]
        for i in range(len(x.shape)-2):
            padding.insert(0, [0,0])
    else: # pad last dim [-1]
        padding = [[0,1]]
        for i in range(len(x.shape)-1):
            padding.insert(0, [0,0])
    y = tf.pad(x, padding, 'symmetric') # symmetric reflect
    return y

### [edited] from https://github.com/aydao/stylegan2-surgery/blob/master/copy_crop_weights.py

def copy_and_crop_or_pad_trainables(src_net, tgt_net) -> None:
    source_trainables = src_net.trainables.keys()
    target_trainables = tgt_net.trainables.keys()
    names = [pair for pair in zip(source_trainables, target_trainables)]
            
    skip = []
    pbar = ProgressBar(len(names))
    for pair in names:
        source_name, target_name = pair
        log = source_name
        x = src_net.get_var(source_name)
        y = tgt_net.get_var(target_name)
        source_shape = x.shape
        target_shape = y.shape
        if source_shape != target_shape:
            update = x
            index = None 
            if 'Dense' in source_name:
                if source_shape[0] > target_shape[0]:
                    gap = source_shape[0] - target_shape[0]
                    start = abs(gap) // 2
                    end = start + target_shape[0]
                    update = update[start:end,:]
                else:
                    update = pad_symm_np(update, target_shape)
                    log = (log, source_shape, '=>', target_shape)
            else:
                try:
                    if source_shape[2] > target_shape[2]:
                        index = 2
                        gap = source_shape[index] - target_shape[index]
                        start = abs(gap) // 2
                        end = start + target_shape[index]
                        update = update[:,:,start:end,:]
                    if source_shape[3] > target_shape[3]:
                        index = 3
                        gap = source_shape[index] - target_shape[index]
                        start = abs(gap) // 2
                        end = start + target_shape[index]
                        update = update[:,:,:,start:end]
                except:
                    print(' Wrong var pair?', source_name, source_shape, target_name, target_shape)
                    exit(1)

                if source_shape[2] < target_shape[2] or source_shape[3] < target_shape[3]:
                    update = pad_symm_np(update, target_shape[2:])
                    log = (log, source_shape, '=>', target_shape)
                    # print(pair, source_shape, target_shape)

            tgt_net.set_var(target_name, update)
            skip.append(source_name)
        pbar.upd(pair)

    weights_to_copy = {tgt_net.vars[pair[1]]: src_net.vars[pair[0]] for pair in names if pair[0] not in skip}
    tfutil.set_vars(tfutil.run(weights_to_copy))

def pad_symm_np(x, size):
    sh = x.shape[-len(size):]
    padding = [[0,0]] * (len(x.shape)-len(size))
    for i, s in enumerate(size):
        p0 = (s-sh[i]) // 2
        p1 = s-sh[i] - p0
        padding.append([p0,p1])
    return np.pad(x, padding, mode='symmetric')

def copy_and_fill_trainables(src_net, tgt_net) -> None: # model => conditional 
    train_vars = [name for name in src_net.trainables.keys() if name in tgt_net.trainables.keys()]
    skip = []
    pbar = ProgressBar(len(train_vars))
    for name in train_vars:
        x = src_net.get_var(name)
        y = tgt_net.get_var(name)
        src_shape = x.shape
        tgt_shape = y.shape
        if src_shape != tgt_shape:
            assert len(src_shape) == len(tgt_shape), "Different shapes: %s %s" % (str(src_shape), str(tgt_shape))
            if np.less(tgt_shape, src_shape).any(): # kill labels: [1024,512] => [512,512]
                try:
                    update = x[:tgt_shape[0], :tgt_shape[1], ...] # !!! corrects only first two dims
                except:
                    update = x[:tgt_shape[0]]
            elif np.greater(tgt_shape, src_shape).any(): # add labels: [512,512] => [1024,512]
                tile_count = [tgt_shape[i] // src_shape[i] for i in range(len(src_shape))]
                if a.verbose is True: print(name, tile_count, src_shape, '=>', tgt_shape, '\n\n') # G_mapping/Dense0, D/Output
                update = np.tile(x, tile_count)
            tgt_net.set_var(name, update)
            skip.append(name)
        pbar.upd(name)
    weights_to_copy = {tgt_net.vars[name]: src_net.vars[name] for name in train_vars if name not in skip}
    tfutil.set_vars(tfutil.run(weights_to_copy))


def main():
    tflib.init_tf({'allow_soft_placement':True})

    G_in, D_in, Gs_in = load_pkl(a.source)
    print(' Loading model', a.source, Gs_in.output_shape)
    _, res_in, _  = calc_init_res(Gs_in.output_shape[1:])
    
    if a.res is not None or a.alpha is True:
        if a.res is None: a.res = Gs_in.output_shape[2:]
        colors = 4 if a.alpha is True else Gs_in.output_shape[1] # EXPERIMENTAL
        _, res_out, _ = calc_init_res([colors, *a.res])

        if res_in != res_out or a.alpha is True: # add or remove layers
            assert G_in is not None and D_in is not None, " !! G/D subnets not found in source model !!"
            data_shape = [colors, res_out, res_out]
            print(' Reconstructing full model with shape', data_shape)
            G_out, D_out, Gs_out = create_model(data_shape, True, 0, Gs_in.static_kwargs)
            copy_vars(Gs_in, Gs_out)
            copy_vars(G_in,  G_out)
            copy_vars(D_in,  D_out, D=True)
            G_in, D_in, Gs_in = G_out, D_out, Gs_out
            a.full = True

        if a.res[0] != res_out or a.res[1] != res_out: # crop or pad layers
            data_shape = [colors, *a.res]
            G_out, D_out, Gs_out = create_model(data_shape, True, 0, Gs_in.static_kwargs)
            if G_in is not None and D_in is not None:
                print(' Reconstructing full model with shape', data_shape)
                copy_and_crop_or_pad_trainables(G_in, G_out)
                copy_and_crop_or_pad_trainables(D_in, D_out)
                G_in, D_in = G_out, D_out
                a.full = True
            else:
                print(' Reconstructing Gs model with shape', data_shape)
            copy_and_crop_or_pad_trainables(Gs_in, Gs_out)
            Gs_in = Gs_out

    if a.labels is not None:
        assert G_in is not None and D_in is not None, " !! G/D subnets not found in source model !!"
        print(' Reconstructing full model with labels', a.labels)
        data_shape = Gs_in.output_shape[1:]
        G_out, D_out, Gs_out = create_model(data_shape, True, a.labels, Gs_in.static_kwargs)
        if a.verbose is True: D_out.print_layers()
        if a.verbose is True: G_out.print_layers()
        copy_and_fill_trainables(G_in, G_out)
        copy_and_fill_trainables(D_in, D_out)
        copy_and_fill_trainables(Gs_in, Gs_out)
        a.full = True

    if a.labels is None and a.res is None and a.alpha is not True:
        if a.reconstruct is True:
            print(' Reconstructing model with same size /', 'full' if a.full else 'Gs')
            data_shape = Gs_in.output_shape[1:]
            G_out, D_out, Gs_out = create_model(data_shape, a.full, 0, Gs_in.static_kwargs)
            Gs_out.copy_vars_from(Gs_in)
            if a.full is True and G_in is not None and D_in is not None:
                G_out.copy_vars_from(G_in)
                D_out.copy_vars_from(D_in)
        else:
            Gs_out = Gs_in

    out_name = basename(a.source)
    if a.res is not None: out_name += '-%dx%d' % (a.res[1], a.res[0])
    if a.alpha is True:   out_name += 'a'
    if a.labels is not None: out_name += '-c%d' % a.labels
        
    if a.full is True: # G_in is not None and D_in is not None
        save_pkl((G_out, D_out, Gs_out), os.path.join(a.out_dir, '%s.pkl' % out_name))
    else:
        save_pkl(Gs_out, os.path.join(a.out_dir, '%s-Gs.pkl' % out_name))

    print(' Done')


if __name__ == '__main__':
    main()
