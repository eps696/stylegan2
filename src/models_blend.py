"""
from https://github.com/justinpinkney/stylegan2/blob/master/blend_models.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import argparse
import math
import numpy as np
import pickle
from imageio import imsave

import tensorflow; tf = tensorflow.compat.v1 if hasattr(tensorflow.compat, 'v1') else tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil

from eps.data_load import basename

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='_out', help='Output directory')
parser.add_argument('--pkl1', required=True, help='PKL for low res layers')
parser.add_argument('--pkl2', required=True, help='PKL for hi res layers')
parser.add_argument('--res', type=int, default=128, help='Resolution level at which to switch between models')
parser.add_argument('--level', type=int, default=0, help='Switch at Conv block 0 or 1?')
parser.add_argument('--blend_width', type=float, default=None, help='None = hard switch, float = smooth switch (logistic) with given width')
parser.add_argument('--verbose', action='store_true', help='Print out the exact blending fraction')
a = parser.parse_args()

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        Gs = pickle.load(f, encoding='latin1')
    try: _, _, Gs = Gs
    except: pass
    return Gs
    
def save_pkl(networks, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(networks, file, protocol=pickle.HIGHEST_PROTOCOL)

def extract_conv_names(model):
    # layers are G_synthesis/{res}x{res}/...
    # make a list of (name, resolution, level, position)
    model_names = list(model.trainables.keys())
    conv_names = []

    resolutions =  [4*2**x for x in range(9)]
    level_names = [["Conv0_up", "Const"], ["Conv1", "ToRGB"]]
    
    position = 0
    # option not to split levels
    for res in resolutions:
        root_name = "G_synthesis/{}x{}/".format(res, res)
        for level, level_suffixes in enumerate(level_names):
            for suffix in level_suffixes:
                search_name = root_name + suffix
                matched_names = [x for x in model_names if x.startswith(search_name)]
                to_add = [(name, "{}x{}".format(res, res), level, position) for name in matched_names]
                conv_names.extend(to_add)
            position += 1

    return conv_names

def main():
    os.makedirs(a.out_dir, exist_ok=True)

    tflib.init_tf()
    with tf.Session() as sess, tf.device('/gpu:0'):
        Gs_lo = load_pkl(a.pkl1)
        Gs_hi = load_pkl(a.pkl2)

        # TODO add small x offset for smoother blend animations
        resolution = "{}x{}".format(a.res, a.res)
        
        model_1_names = extract_conv_names(Gs_lo)
        model_2_names = extract_conv_names(Gs_hi)
        assert all((x == y for x, y in zip(model_1_names, model_2_names)))

        Gs_out = Gs_lo.clone()
        
        short_names = [(x[1:3]) for x in model_1_names]
        full_names = [(x[0]) for x in model_1_names]
        mid_point_idx = short_names.index((resolution, a.level))
        mid_point_pos = model_1_names[mid_point_idx][3]
        
        ys = []
        for name, resolution, level, position in model_1_names:
            x = position - mid_point_pos
            if a.blend_width is not None:
                exponent = -x / a.blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 1 else 0
            ys.append(y)
            if a.verbose is True:
                print("Blending {} by {}".format(name, y))

        tfutil.set_vars(tfutil.run({ 
                 Gs_out.vars[name]: (Gs_hi.vars[name] * y + Gs_lo.vars[name] * (1-y))
                 for name, y in zip(full_names, ys)} ))

        out_name = os.path.join(a.out_dir, '%s-%s-%d-%d' % (basename(a.pkl1).split('-')[0], basename(a.pkl2).split('-')[0], a.res, a.level))  
        save_pkl(Gs_out, '%s.pkl' % out_name)
            
        rnd = np.random.RandomState(696)
        grid_latents = rnd.randn(4, *Gs_lo.input_shape[1:])
        grid_fakes = Gs_out.run(grid_latents, [None], is_validation=True, minibatch_size=1)
        grid_fakes = np.hstack(np.transpose(grid_fakes, [0,2,3,1]))
        imsave('%s.jpg' % out_name, ((grid_fakes+1)*127.5).astype(np.uint8))
        
if __name__ == '__main__':
    main()

