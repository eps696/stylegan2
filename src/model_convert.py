import os
import argparse
import pickle
from collections import OrderedDict
import numpy as np

import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil

from eps.utilgan import calc_init_res
from eps.progress_bar import ProgressBar
from eps.data_load import basename

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Source model path')
parser.add_argument('--out_dir', default='./', help='Output directory for reduced/reconstructed model')
parser.add_argument('--reconstruct', '-r', action='store_true', help='Reconstruct model (add internal arguments)')
parser.add_argument('--res', '-s', default=None, help='Target resolution in format X-Y')
parser.add_argument('--verbose', '-v', action='store_true')
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

def create_model(data_shape, full=False, kwargs_in=None):
    init_res, resolution, res_log2 = calc_init_res(data_shape[1:])
    kwargs_out = dnnlib.EasyDict()
    kwargs_out.num_channels = data_shape[0]
    kwargs_out.label_size = 0
    if kwargs_in is not None:
        for k in list(kwargs_in.keys()):
            kwargs_out[k] = kwargs_in[k]
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

### from https://github.com/tr1pzz/StyleGAN-Resolution-Convertor/blob/master/StyleGAN_resolution_conversion.ipynb

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
            print('.. Renaming -- %s%s to -- %s%s -- shape: %s' % (key, spacing, target_key, spacing, str(model_dict[key].shape)))
            model_dict = OrderedDict([(target_key, v) if k == key else (k, v) for k, v in model_dict.items()])

    subnetwork.vars = model_dict
    subnetwork.own_vars = OrderedDict(subnetwork.vars)
    subnetwork.trainables = OrderedDict((name, var) for name, var in subnetwork.vars.items() if var.trainable)
    subnetwork.var_global_to_local = OrderedDict((var.name.split(":")[0], name) for name, var in subnetwork.vars.items())
    return subnetwork.trainables.copy()

def copy_weights(src_net, tgt_net, vars_to_copy):
    names = [name for name in tgt_net.trainables.keys() if name in vars_to_copy]
    tfutil.set_vars(tfutil.run({tgt_net.vars[name]: src_net.vars[name] for name in names}))

### from https://github.com/aydao/stylegan2-surgery/blob/master/copy_crop_weights.py

def copy_and_crop_trainables(src_net, tgt_net) -> None:
    source_trainables = src_net.trainables.keys()
    target_trainables = tgt_net.trainables.keys()
    names = [pair for pair in zip(source_trainables, target_trainables)]
            
    skip = []
    pbar = ProgressBar(len(names))
    for pair in names:
        source_name, target_name = pair
        x = src_net.get_var(source_name)
        y = tgt_net.get_var(target_name)
        source_shape = x.shape
        target_shape = y.shape
        if source_shape != target_shape:
            update = x
            index = None 
            if 'Dense' in source_name:
                index = 0
                gap = source_shape[index] - target_shape[index]
                start = abs(gap) // 2
                end = start + target_shape[index]
                update = update[start:end,:]
            else:
                try:
                    if source_shape[2] != target_shape[2]:
                        index = 2
                        gap = source_shape[index] - target_shape[index]
                        start = abs(gap) // 2
                        end = start + target_shape[index]
                        update = update[:,:,start:end,:]
                    if source_shape[3] != target_shape[3]:
                        index = 3
                        gap = source_shape[index] - target_shape[index]
                        start = abs(gap) // 2
                        end = start + target_shape[index]
                        update = update[:,:,:,start:end]
                except:
                    print(' Wrong var pair?', source_name, source_shape, target_name, target_shape)
                    exit(1)

            tgt_net.set_var(target_name, update)
            skip.append(source_name)
        pbar.upd(pair)

    weights_to_copy = {tgt_net.vars[pair[1]]: src_net.vars[pair[0]] for pair in names if pair[0] not in skip}
    tfutil.set_vars(tfutil.run(weights_to_copy))


def main():
    tflib.init_tf({'allow_soft_placement':True})

    print(' Loading model', a.source)
    G_in, D_in, Gs_in = load_pkl(a.source)

    if a.res is not None:
        print(' Reconstructing model with size', a.res)
        data_shape = [Gs_in.output_shape[1], *a.res]
        G_out, D_out, Gs_out, res_out_log2 = create_model(data_shape, True, Gs_in.static_kwargs)

        if a.res[0] == a.res[1]:
            assert G_in is not None and D_in is not None, " !! G/D subnets not found in source model !!"
            res_in_log2 = np.log2(get_model_res(Gs_in))
            lod_diff = res_out_log2 - res_in_log2
            Gs_in_names_to_copy = update_dict_keys(Gs_in, 'ToRGB_lod',   'generator',     lod_diff)
            G_in_names_to_copy  = update_dict_keys(G_in,  'ToRGB_lod',   'generator',     lod_diff)
            D_in_names_to_copy  = update_dict_keys(D_in,  'FromRGB_lod', 'discriminator', lod_diff)
            copy_weights(Gs_in, Gs_out, Gs_in_names_to_copy)
            copy_weights(G_in,  G_out,  G_in_names_to_copy)
            copy_weights(D_in,  D_out,  D_in_names_to_copy)
            
        else: # EXPERIMENTAL .. check source repo 
            if G_in is not None and D_in is not None:
                copy_and_crop_trainables(G_in, G_out)
                copy_and_crop_trainables(D_in, D_out)
            copy_and_crop_trainables(Gs_in, Gs_out)

    elif a.reconstruct is True:
        print(' Reconstructing model with same size')
        data_shape = Gs_in.output_shape[1:]
        _, _, Gs_out, _ = create_model(data_shape, False, Gs_in.static_kwargs)
        Gs_out.copy_vars_from(Gs_in)

    else:
        Gs_out = Gs_in

    if a.res is not None: 
        if G_in is not None and D_in is not None:
            save_pkl((G_out, D_out, Gs_out), os.path.join(a.out_dir, '%s-%dx%d.pkl' % (basename(a.source), *a.res[::-1])))
        else:
            save_pkl(Gs_out, os.path.join(a.out_dir, '%s-%dx%d.pkl' % (basename(a.source), *a.res[::-1])))
    else:
        save_pkl(Gs_out, os.path.join(a.out_dir, '%s-Gs.pkl' % basename(a.source)))

    print(' Done')


if __name__ == '__main__':
    main()
