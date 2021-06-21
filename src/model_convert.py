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

from eps.utilgan import calc_init_res
from eps.progress_bar import ProgressBar
from eps.data_load import basename

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Source model path')
parser.add_argument('--out_dir', default='./', help='Output directory for reduced/reconstructed model')
parser.add_argument('-r', '--reconstruct', action='store_true', help='Reconstruct model (add internal arguments)')
parser.add_argument('-s', '--res', default=None, help='Target resolution in format X-Y')
parser.add_argument('-a', '--alpha', action='store_true', help='Add alpha channel for RGBA processing')
parser.add_argument('-l', '--labels', default=0, type=int, help='Make conditional model')
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

def create_model(data_shape, full=False, labels=0, kwargs_in=None):
    init_res, resolution, res_log2 = calc_init_res(data_shape[1:])
    kwargs_out = dnnlib.EasyDict()
    kwargs_out.num_channels = data_shape[0]
    if kwargs_in is not None:
        for k in list(kwargs_in.keys()):
            kwargs_out[k] = kwargs_in[k]
    if labels > 0: kwargs_out.label_size = labels
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
    return G, D, Gs, res_log2

### [edited] from https://github.com/tr1pzz/StyleGAN-Resolution-Convertor/blob/master/StyleGAN_resolution_conversion.ipynb

def get_model_res(G_net):
    try:
        res = G_net.static_kwargs['resolution']
    except:
        res = G_net.output_shape[1]
    return res

def replace_char(start_string, index, new_char):
    new = list(start_string)
    new[index] = new_char
    return ''.join(new)

def update_dict_keys(subnetwork, filter_string, model_name, lod_diff):
    model_dict = subnetwork.vars
    if model_name == 'discriminator':
        model_dict = OrderedDict(reversed(list(model_dict.items())))
    
    for i, key in enumerate(model_dict):
        if filter_string in key:
            index = key.find(filter_string) + len(filter_string)
            lod_str = str(int(key[index]) + lod_diff)
            target_key = replace_char(key, index, lod_str)
            spacing = '    ' if 'bias' in key else ''
            if a.verbose is True: print('.. Renaming -- %s%s to -- %s%s -- shape: %s' % (key, spacing, target_key, spacing, str(model_dict[key].shape)))
            model_dict = OrderedDict([(target_key, v) if k == key else (k, v) for k, v in model_dict.items()])

    subnetwork.vars = model_dict
    subnetwork.own_vars = OrderedDict(subnetwork.vars)
    subnetwork.trainables = OrderedDict((name, var) for name, var in subnetwork.vars.items() if var.trainable)
    subnetwork.var_global_to_local = OrderedDict((var.name.split(":")[0], name) for name, var in subnetwork.vars.items())
    return subnetwork.trainables.copy()

def copy_weights(src_net, tgt_net, vars_to_copy, D=False):
    names = [name for name in tgt_net.trainables.keys() if name in vars_to_copy]
    var_dict = OrderedDict()

    for name in names:
        if tgt_net.vars[name].shape == src_net.vars[name].shape: # fixing rgb-to-rgba only !!
            var_dict[name] = src_net.vars[name]
        else:
            var_dict[name] = add_channel(src_net.vars[name], D=D)
            # print(name)

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
    source_trainables = src_net.trainables.keys()
    skip = []
    pbar = ProgressBar(len(source_trainables))
    for name in source_trainables:
        x = src_net.get_var(name)
        y = tgt_net.get_var(name)
        source_shape = x.shape
        target_shape = y.shape
        if source_shape != target_shape:
            assert len(source_shape) == len(target_shape), "Different shapes: %s %s" % (str(source_shape), str(target_shape))
            tile_count = [target_shape[i] // source_shape[i] for i in range(len(source_shape))]
            if a.verbose is True: print(name, tile_count, source_shape, '=>', target_shape, '\n\n') # G_mapping/Dense0, D/Output
            update = np.tile(x, tile_count) # [512,512] => [1024,512]
            tgt_net.set_var(name, update)
            skip.append(name)
        pbar.upd(name)
    weights_to_copy = {tgt_net.vars[name]: src_net.vars[name] for name in source_trainables if name not in skip}
    tfutil.set_vars(tfutil.run(weights_to_copy))


def main():
    tflib.init_tf({'allow_soft_placement':True})

    G_in, D_in, Gs_in = load_pkl(a.source)
    print(' Loading model', a.source, Gs_in.output_shape)
    _, res_in, _  = calc_init_res(Gs_in.output_shape[1:])
    save_full = False
    
    if a.res is not None or a.alpha is True:
        if a.res is None: a.res = Gs_in.output_shape[2:]
        colors = 4 if a.alpha is True else Gs_in.output_shape[1] # EXPERIMENTAL
        _, res_out, _ = calc_init_res([colors, *a.res])

        if res_in != res_out or a.alpha is True: # add or remove layers
            assert G_in is not None and D_in is not None, " !! G/D subnets not found in source model !!"
            data_shape = [colors, res_out, res_out]
            print(' Reconstructing full model with shape', data_shape)
            G_out, D_out, Gs_out, res_out_log2 = create_model(data_shape, True, 0, Gs_in.static_kwargs)
            res_in_log2 = np.log2(get_model_res(Gs_in))
            lod_diff = res_out_log2 - res_in_log2
            Gs_in_names_to_copy = update_dict_keys(Gs_in, 'ToRGB_lod',   'generator',     lod_diff)
            G_in_names_to_copy  = update_dict_keys(G_in,  'ToRGB_lod',   'generator',     lod_diff)
            D_in_names_to_copy  = update_dict_keys(D_in,  'FromRGB_lod', 'discriminator', lod_diff)
            copy_weights(Gs_in, Gs_out, Gs_in_names_to_copy)
            copy_weights(G_in,  G_out,  G_in_names_to_copy)
            copy_weights(D_in,  D_out,  D_in_names_to_copy, D=True)
            G_in, D_in, Gs_in = G_out, D_out, Gs_out
            save_full = True

        if a.res[0] != res_out or a.res[1] != res_out: # crop or pad layers
            data_shape = [colors, *a.res]
            print(' Reconstructing model with shape', data_shape)
            G_out, D_out, Gs_out, res_out_log2 = create_model(data_shape, True, 0, Gs_in.static_kwargs)
            if G_in is not None and D_in is not None:
                copy_and_crop_or_pad_trainables(G_in, G_out)
                copy_and_crop_or_pad_trainables(D_in, D_out)
                G_in, D_in = G_out, D_out
            copy_and_crop_or_pad_trainables(Gs_in, Gs_out)
            Gs_in = Gs_out

    if a.labels > 0:
        assert G_in is not None and D_in is not None, " !! G/D subnets not found in source model !!"
        print(' Reconstructing full model with labels', a.labels)
        data_shape = Gs_in.output_shape[1:]
        G_out, D_out, Gs_out, _ = create_model(data_shape, True, a.labels, Gs_in.static_kwargs)
        if a.verbose is True: D_out.print_layers()
        if a.verbose is True: G_out.print_layers()
        copy_and_fill_trainables(G_in, G_out)
        copy_and_fill_trainables(D_in, D_out)
        copy_and_fill_trainables(Gs_in, Gs_out)
        save_full = True

    if a.labels == 0 and a.res is None and a.alpha is not True:
        if a.reconstruct is True:
            print(' Reconstructing model with same size')
            data_shape = Gs_in.output_shape[1:]
            _, _, Gs_out, _ = create_model(data_shape, False, 0, Gs_in.static_kwargs)
            Gs_out.copy_vars_from(Gs_in)
        else:
            Gs_out = Gs_in

    out_name = basename(a.source)
    if a.res is not None: out_name += '-%dx%d' % (a.res[1], a.res[0])
    if a.alpha is True:   out_name += 'a'
    if a.labels > 0:      out_name += '-c%d' % a.labels
        
    if save_full is True: # G_in is not None and D_in is not None
        save_pkl((G_out, D_out, Gs_out), os.path.join(a.out_dir, '%s.pkl' % out_name))
    else:
        save_pkl(Gs_out, os.path.join(a.out_dir, '%s-Gs.pkl' % out_name))

    print(' Done')


if __name__ == '__main__':
    main()
