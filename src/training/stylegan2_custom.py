# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
# This work is made available under the Nvidia Source Code License-NC.
# https://nvlabs.github.io/stylegan2/license.html
"""Network architectures used in the StyleGAN2 paper."""

import numpy as np
import tensorflow as tf

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d

from .networks_stylegan2 import G_mapping, conv2d_layer, apply_bias_act, modulated_conv2d_layer

from util.utilgan import hw_scales, fix_size

# Main generator network -- same as original
def G_main(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.5,                      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                      # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',              # Build func name for the mapping network.
    synthesis_func          = 'G_synthesis_stylegan2',  # Build func name for the synthesis network.
    **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            # original version
            mixing_cutoff = tf.cond( 
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            """ # Diff Augment version
            mixing_cutoff = tf.where_v2(
                tf.random_uniform([tf.shape(dlatents)[0]], 0.0, 1.0) < style_mixing_prob,
                tf.random_uniform([tf.shape(dlatents)[0]], 1, cur_layers, dtype=tf.int32),
                cur_layers[np.newaxis])[:, np.newaxis, np.newaxis]
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
            """

    # Apply truncation trick.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out = components.synthesis.get_output_for(dlatents, is_training=is_training, force_clean_graph=is_template_graph, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out

# StyleGAN2 synthesis network. Implements skip connections and residual nets (Figure 7), but no progressive growing.
def G_synthesis_stylegan2(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Base model resolution (corresponding to the layer count)
    init_res            = [4,4],      # Initial (minimal) resolution for progressive training
    size                = None,       # Output size
    scale_type          = None,       # Arbitrary size: fit (scaling), reflect (pad), symmetric (pad)
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    verbose             = False,      #
    impl                = 'cuda',     # Custom ops implementation - cuda (original) or ref (no compiling)
    **_kwargs):                         # Ignore unrecognized keyword args.

    res_log2 = int(np.log2(resolution))
    assert resolution == 2**res_log2 and resolution >= 4
    
    # calculate intermediate layers sizes for arbitrary output resolution
    custom_res = (resolution * init_res[0] // 4, resolution * init_res[1] // 4)
    if size is None: size = custom_res
    if init_res != [4,4] and verbose:
        print(' .. init res', init_res, size)
    keep_first_layers = 2 if scale_type == 'fit' else None
    hws = hw_scales(size, custom_res, res_log2 - 2, keep_first_layers, verbose)
    if verbose: print(hws, '..', custom_res, res_log2-1)
    
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity
    num_layers = res_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**(res-2) * init_res[0], 2**(res-2) * init_res[1]]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, size, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv, impl=impl)
        if size is not None and up is True:
            x = fix_size(x, size, scale_type)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise = fix_size(noise, (x.shape[2], x.shape[3]), scale_type=scale_type)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act, impl=impl)

    # Building blocks for main layers.
    def block(x, res, size): # res = 3..res_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res*2-5, size=size, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res*2-4, size=size, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel, impl=impl)
                if size is not None:
                    t = fix_size(t, (x.shape[2], x.shape[3]), scale_type=scale_type)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel, impl=impl)

    def torgb(x, y, res): # res = 2..res_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv, impl=impl), impl=impl)
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), *init_res], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, size=None, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)

    # Main layers.
    for res in range(3, res_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(x, res, hws[res-2])
            if architecture == 'skip':
                y = upsample(y)
                if size is not None:
                    y = fix_size(y, hws[res-2], scale_type=scale_type)
            if architecture == 'skip' or res == res_log2:
                y = torgb(x, y, res)
    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')

