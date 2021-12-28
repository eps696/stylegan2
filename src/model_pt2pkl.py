import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings("ignore")
import sys
import math
import argparse
import numpy as np
import pickle

import torch
from torchvision import utils

# sys.path.append(args.dnnlib)
import dnnlib
from dnnlib import tflib

try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

parser = argparse.ArgumentParser(description="Rosinality (pytorch) to Nvidia (pkl) checkpoint converter")
parser.add_argument("--model_pkl", metavar="PATH", help="path to the source tensorflow weights")
parser.add_argument("--model_pt", metavar="PATH", help="path to the updated pytorch weights")
args = parser.parse_args()

def save_pkl(networks, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(networks, file, protocol=pickle.HIGHEST_PROTOCOL)

def update(network, name, value):
    network.set_var(name, value.cpu().numpy())

def convert_modconv(tgt_net, src_dict, target_name, source_name, flip=False):
    conv_weight = src_dict[source_name + ".conv.weight"].squeeze(0).permute(2,3,1,0)
    if flip:
        conv_weight = torch.flip(conv_weight, [0,1])
    update(tgt_net, target_name + "/weight", conv_weight)
    update(tgt_net, target_name + "/mod_weight",     src_dict[source_name + ".conv.modulation.weight"].permute(1,0))
    update(tgt_net, target_name + "/mod_bias",       src_dict[source_name + ".conv.modulation.bias"] - 1.)
    update(tgt_net, target_name + "/noise_strength", src_dict[source_name + ".noise.weight"].squeeze())
    update(tgt_net, target_name + "/bias",           src_dict[source_name + ".activate.bias"].squeeze())

def convert_torgb(tgt_net, src_dict, target_name, source_name):
    update(tgt_net, target_name + "/weight",     src_dict[source_name + ".conv.weight"].squeeze(0).permute(2,3,1,0))
    update(tgt_net, target_name + "/mod_weight", src_dict[source_name + ".conv.modulation.weight"].permute(1,0))
    update(tgt_net, target_name + "/mod_bias",   src_dict[source_name + ".conv.modulation.bias"] - 1.)
    update(tgt_net, target_name + "/bias",       src_dict[source_name + ".bias"].squeeze())

def convert_dense(tgt_net, src_dict, target_name, source_name):
    update(tgt_net, target_name + "/weight", src_dict[source_name + ".weight"].permute(1,0))
    update(tgt_net, target_name + "/bias",   src_dict[source_name + ".bias"])

def update_G(src_dict, tgt_net, size, n_mlp):
    log_size = int(math.log(size, 2))

    pbar = ProgressBar(n_mlp + log_size-2 + log_size-2 + (log_size-2)*2+1 + 2)
    for i in range(n_mlp):
        convert_dense(tgt_net, src_dict, f"G_mapping/Dense{i}", f"style.{i+1}")
        pbar.upd()
    update(tgt_net, "G_synthesis/4x4/Const/const", src_dict["input.input"])
    convert_torgb(tgt_net, src_dict, "G_synthesis/4x4/ToRGB", "to_rgb1")
    pbar.upd()

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        convert_torgb(tgt_net, src_dict, f"G_synthesis/{reso}x{reso}/ToRGB", f"to_rgbs.{i}")
        pbar.upd()
    convert_modconv(tgt_net, src_dict, "G_synthesis/4x4/Conv", "conv1")
    pbar.upd()

    conv_i = 0
    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        convert_modconv(tgt_net, src_dict, f"G_synthesis/{reso}x{reso}/Conv0_up", f"convs.{conv_i}", flip=True)
        convert_modconv(tgt_net, src_dict, f"G_synthesis/{reso}x{reso}/Conv1", f"convs.{conv_i + 1}")
        conv_i += 2
        pbar.upd()

    for i in range(0, (log_size - 2) * 2 + 1):
        update(tgt_net, f"G_synthesis/noise{i}", src_dict[f"noises.noise_{i}"])
        pbar.upd()


if __name__ == "__main__":
    tflib.init_tf()

    with open(args.model_pkl, "rb") as f:
        nets = pickle.load(f)
        try:
            G_in, D_in, Gs = nets
        except:
            Gs = nets

    src_dict = torch.load(args.model_pt)
    
    mapping_layers = Gs.components.mapping.list_layers()
    n_mlp = len([l for l in mapping_layers if l[0].startswith('Dense')])
    size = Gs.output_shape[2] # 1024

    update_G(src_dict['g_ema'], Gs, size, n_mlp)

    out_name = args.model_pt.replace('.pt', '.pkl')
    save_pkl(Gs, out_name)

