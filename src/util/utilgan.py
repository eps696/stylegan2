import os
import sys
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline as CubSpline
from scipy.special import comb
import scipy
from skimage.transform import resize as imresize 
from imageio import imread, imsave

import tensorflow; tf = tensorflow.compat.v1 if hasattr(tensorflow.compat,'v1') else tensorflow

# from perlin import PerlinNoiseFactory as Perlin
# noise = Perlin(1)

# def latent_noise(t, dim, noise_step=78564.543):
    # latent = np.zeros((1, dim))
    # for i in range(dim):
        # latent[0][i] = noise(t + i * noise_step)
    # return latent

def load_latents(npy_file):
    key_latents = np.load(npy_file)
    idx_file = os.path.splitext(npy_file)[0] + '.txt'
    if os.path.exists(idx_file): 
        with open(idx_file) as f:
            lat_idx = f.read()
            lat_idx = [int(l) for l in lat_idx.split(',')]
        key_latents = [key_latents[i] for i in lat_idx]
    return np.asarray(key_latents)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_z(shape, seed=None, uniform=False):
    if seed is None:
        seed = np.random.seed(int((time.time()%1) * 9999))
    rnd = np.random.RandomState(seed)
    if uniform:
        return rnd.uniform(0., 1., shape)
    else:
        return rnd.randn(*shape) # *x unpacks tuple/list to sequence

def smoothstep(x, NN=1., xmin=0., xmax=1.):
    N = math.ceil(NN)
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, N+1):
         result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
    result *= x**(N+1)
    if NN != N: result = (x + result) / 2
    return result

def lerp(z1, z2, num_steps, smooth=0.): 
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interpol = z1 + (z2 - z1) * x
        vectors.append(interpol)
    return np.array(vectors)

# interpolate on hypersphere
def slerp(z1, z2, num_steps, smooth=0.):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = np.linalg.norm(interp)
        interpol_normal = interplain * (z1_norm / interp_norm)
        # interpol_normal = interp * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return np.array(vectors)

def cublerp(points, steps, fstep):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    points = np.concatenate((points, np.expand_dims(points[0], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def latent_anima(shape, frames, transit, key_latents=None, smooth=0.5, cubic=False, gauss=False, seed=None, verbose=True):
    if key_latents is None:
        transit = int(max(1, min(frames//4, transit)))
    steps = max(1, int(frames // transit))
    log = ' timeline: %d steps by %d' % (steps, transit)

    getlat = lambda : get_z(shape, seed=seed)
    
    # make key points
    if key_latents is None:
        key_latents = np.array([getlat() for i in range(steps)])

    latents = np.expand_dims(key_latents[0], 0)
    
    # populate lerp between key points
    if transit == 1:
        latents = key_latents
    else:
        if cubic:
            latents = cublerp(key_latents, steps, transit)
            log += ', cubic'
        else:
            for i in range(steps):
                zA = key_latents[i]
                zB = key_latents[(i+1) % steps]
                interps_z = slerp(zA, zB, transit, smooth=smooth)
                latents = np.concatenate((latents, interps_z))
    latents = np.array(latents)
    
    if gauss:
        lats_post = gaussian_filter(latents, [transit, 0, 0], mode="wrap")
        lats_post = (lats_post / np.linalg.norm(lats_post, axis=-1, keepdims=True)) * math.sqrt(np.prod(shape))
        log += ', gauss'
        latents = lats_post
        
    if verbose: print(log)
    if latents.shape[0] > frames: # extra frame
        latents = latents[1:]
    return latents
    
# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def ups2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

def pad_up_to(x, size, type='centr'):
    sh = (x.shape[1].value, x.shape[2].value)
    padding = [[0,0]]
    for i, s in enumerate(size):
        if type.lower() == 'centr':
            p0 = (s-sh[i]) // 2
            p1 = s-sh[i] - p0
            padding.append([p0, p1])
        else:
            padding.append([0, s-sh[i]])
    padding.append([0,0])
    y = tf.pad(x, padding, 'symmetric') # REFLECT, SYMMETRIC
    # print(' pad', sh, 'to', size, '\t == ', padding, y.shape)
    return y

def fix_size_np(x, size): # for dataset to tfrecords
    if not len(x.shape) == 3:
        raise Exception(" Wrong data rank, shape:", x.shape)
    if (x.shape[1], x.shape[2]) == size:
        return x
    x = x.transpose(1,2,0)
    x = imresize(x, size, preserve_range=True)
    x = x.transpose(2,0,1)
    return x

def fix_size(x, size, scale_type='centr', order='BCHW'): # scale_type = one of [fit, centr, side]
    if not len(x.get_shape()) == 4:
        raise Exception(" Wrong data rank, shape:", x.get_shape())
    if (x.get_shape()[2], x.get_shape()[3]) == size:
        return x
    if (x.get_shape()[2]*2, x.get_shape()[3]*2) == size:
        return ups2d(x)
    if order == 'BCHW': # BCHW for PGAN/SGANs, BHWC for old GANs
        x = tf.transpose(x, [0,2,3,1])
    if scale_type.lower() == 'fit':
        x = tf.image.resize_images(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    else: # proportional scale to smaller side, then pad to bigger side
        sh0 = x.get_shape().as_list()[1:3]
        try: size = [s.value for s in size]
        except: pass
        upsc = np.min([float(size[i]) / float(sh0[i]) for i in [0,1]])
        new_size = [int(sh0[i]*upsc) for i in [0,1]]
        x = tf.image.resize_images(x, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        # workaround if pad > srcsize (i.e. not enough pixels to fill in)
        for i in [0,1]:
            while size[i] > 2*new_size[i]:
                new_size[i] *= 2
                x = pad_up_to(x, new_size, scale_type)
        x = pad_up_to(x, size, scale_type)
    if order == 'BCHW': # BCHW for PGAN/SGANs, BHWC for old GANs
        x = tf.transpose(x, [0,3,1,2])
    return x

# Make list of odd sizes for upsampling to arbitrary resolution
def hw_scales(size, base, n, keep_first_layers=None, verbose=False):
    if isinstance(base, int): base = (base, base)
    start_res = [int(b * 2 ** (-n)) for b in base]
    
    start_res[0] = start_res[0] * size[0] // base[0]
    start_res[1] = start_res[1] * size[1] // base[1]

    hw_list = []
    
    if base[0] != base[1] and verbose is True:
        print(' size', size, 'base', base, 'start_res', start_res, 'n', n)
    if keep_first_layers is not None and keep_first_layers > 0:
        for i in range(keep_first_layers):
            hw_list.append(start_res)
            start_res = [x * 2 for x in start_res]
            n -= 1
            
    ch = (size[0] / start_res[0]) ** (1/n)
    cw = (size[1] / start_res[1]) ** (1/n)
    for i in range(n):
        h = math.floor(start_res[0] * ch**i)
        w = math.floor(start_res[1] * cw**i)
        hw_list.append((h,w))

    hw_list.append(size)
    return hw_list

def calc_res(shape):
    base0 = 2**int(np.log2(shape[0]))
    base1 = 2**int(np.log2(shape[1]))
    base = min(base0, base1)
    min_res = min(shape[0], shape[1])
    if min_res != base or max(*shape) / min(*shape) >= 2:
        if np.log2(base) < 10:
            base = base * 2
    return base # , [shape[0]/base, shape[1]/base]

def calc_init_res(shape, resolution=None):
    if len(shape) == 1:
        shape = [shape[0], shape[0], 1]
    elif len(shape) == 2:
        shape = [*shape, 1]
    size = shape[:2] if shape[2] < min(*shape[:2]) else shape[1:] # fewer colors than pixels
    if resolution is None:
        resolution = calc_res(size)
    res_log2 = int(np.log2(resolution))
    init_res = [int(s * 2**(2-res_log2)) for s in size]
    return init_res, resolution, res_log2

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(in_dir, ext=None):
    all_files = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    if ext is not None: 
        if isinstance(ext, list):
            files = []
            for e in ext:
                files += [f for f in all_files if f.endswith(e)]
        elif isinstance(ext, str):
            files = [f for f in all_files if f.endswith(ext)]
        else:
            files = all_files
    return sorted([f for f in files if os.path.isfile(f)])

def dir_list(in_dir):
    dirs = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    return sorted([f for f in dirs if os.path.isdir(f)])

def img_list(path):
    files = [os.path.join(path, file_i) for file_i in os.listdir(path)
            if file_i.lower().endswith('.jpg') or file_i.lower().endswith('.jpeg') or file_i.lower().endswith('.png')]
    return sorted([f for f in files if os.path.isfile(f)])

def img_read(path):
    img = imread(path)
    # 8bit to 256bit
    if (img.ndim == 2) or (img.shape[2] == 1):
        img = np.dstack((img,img,img))
    # rgba to rgb 
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img
    
