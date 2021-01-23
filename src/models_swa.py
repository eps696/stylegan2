"""
https://github.com/arfafax/StyleGAN2_experiments
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
See: https://github.com/kristpapadopoulos/keras-stochastic-weight-averaging
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import pickle
import argparse

import dnnlib
import dnnlib.tflib as tflib

parser = argparse.ArgumentParser(description='Perform stochastic weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dir', default='models', help='Directory with network checkpoints for weight averaging')
parser.add_argument('--output', default='output-mixed.pkl', help='The averaged model to output')
parser.add_argument('--count', default=None, help='Average the last n checkpoints', type=int)
# args, other_args = parser.parse_known_args()
args = parser.parse_args()

def fetch_models_from_files(model_list):
    for fn in model_list:
        with open(fn, 'rb') as f:
            yield pickle.load(f)

def apply_swa_to_checkpoints(models):
    next_models = next(models)
    try:
        gen, dis, gs = next_models
        mod_gen, mod_dis, mod_gs = gen, dis, gs
    except:
        gs = next_models
        mod_gs = gs
    print('Loading 1 ', end='', flush=True)
    epoch = 0
    try:
        while True:
            epoch += 1
            print('. ', end='', flush=True)
            next_models = next(models)
            if next_models is None: 
                print('')
                break
            try:
                gen, dis, gs = next_models
                # mod_gen.apply_swa(gen, epoch)
                # mod_dis.apply_swa(dis, epoch)
            except:
                gs = next_models
            mod_gs.apply_swa(gs, epoch)
            print('%d '%(epoch+1), end='', flush=True)
    except:
        print("end")
    # return (mod_gen, mod_dis, mod_gs)
    return mod_gs

def main():
    files = glob.glob(os.path.join(args.in_dir, '*.pkl'))
    files.sort()
    if args.count is not None:
        if (len(files) > args.count):
            files = files[-args.count:]
    print(files)
    
    tflib.init_tf()
    models = fetch_models_from_files(files)
    swa_models = apply_swa_to_checkpoints(models)

    with open(args.output, 'wb') as f:
        pickle.dump(swa_models, f)
    print('Output model with stochastic averaged weights saved as', args.output)

main()