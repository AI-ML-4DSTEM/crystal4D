import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

#Custom libraries - User defined
from ..layers import convSpectral
from ..layers import poolSpectral

'''
These functions are to create the custom layers
'''

# %%
#Adapted from get_cross_correlation algorithm in py4DStem (corrPower = 0)

def cross_correlate(x):
    cb = x[0]
    pr = x[1]
    cb = tf.keras.backend.permute_dimensions(cb, (3,0,1,2))
    pr = tf.keras.backend.permute_dimensions(pr, (3,0,1,2))
    
    #shift probe to the origin
    pr = tf.signal.ifftshift(pr, axes = (2,3))
    cb = tf.signal.ifftshift(cb, axes = (2,3))
    cbed = tf.signal.fft2d(tf.cast(cb,tf.complex64))
    probe = tf.signal.fft2d(tf.cast(pr,tf.complex64))
    
    ccff = tf.multiply(cbed, tf.math.conj(probe))
    cc = tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(ccff)))
    cc = tf.keras.backend.permute_dimensions(cc, (1,2,3,0))
    
    #normalize each cross-corr
    cc, _ = tf.linalg.normalize(cc, axis = (1,2))                 
        
    return cc


def cross_correlate_ff(x):
    cb = x[0]
    pr = x[1]
    cb = tf.keras.backend.permute_dimensions(cb, (3,0,1,2))
    pr = tf.keras.backend.permute_dimensions(pr, (3,0,1,2))
    
    #shift probe to the origin
    pr = tf.signal.ifftshift(pr, axes = (2,3))
    cb = tf.signal.ifftshift(cb, axes = (2,3))
    cbed = tf.signal.fft2d(tf.cast(cb,tf.complex64))
    probe = tf.signal.fft2d(tf.cast(pr,tf.complex64))
    
    ccff = tf.multiply(cbed, tf.math.conj(probe))
    ccff = tf.signal.fftshift(ccff, axes = (2,3))
    ccff = tf.keras.backend.permute_dimensions(ccff, (1,2,3,0))
    
    #normalize each cross-corr
    ccff, _ = tf.linalg.normalize(ccff, axis = (1,2)) 
    ccff_real = tf.math.real(ccff)
    ccff_im = tf.math.imag(ccff)
    ccff = tf.concat([ccff_real,ccff_im], axis = -1)
    
    return ccff


def cross_correlate_iff(x):
    input_channel = x.shape[-1] // 2

    input_complex = tf.dtypes.complex(x[:,:,:,:input_channel], x[:,:,:,input_channel:])
    input_transposed = tf.transpose(input_complex, [0,3,1,2])
    input_transposed = tf.signal.fftshift(input_transposed, axes = (2,3))
    output_complex = tf.math.real(tf.signal.ifftshift(tf.signal.ifft2d(input_transposed), axes = (2,3)))
    output = tf.transpose(output_complex, [0,2,3,1])
    
    return output

# %%
#MCDropout
class MonteCarloDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

def conv2d_block(input_tensor, n_filters, n_depth=2, input_channel=1, kernel_size = 3, activation = 'relu', dp_rate = 0.1, batchnorm = True):
    """Function to add n (depth) convolutional layers with the parameters passed to it"""
    
    kl = 'he_uniform'
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(input_tensor)
    else:
        x = input_tensor
    
    for _ in range(n_depth):
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),             
                               padding = 'same', kernel_initializer = kl, bias_initializer = kl, kernel_regularizer= 'l2', bias_regularizer= 'l2')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = MonteCarloDropout(dp_rate)(x)
    
    return x

def convSpec2d_block(input_tensor, n_filters, n_depth=2, input_channel=1, kernel_size = 3, activation = 'relu', dp_rate = 0.1, batchnorm = True):
    """Function to add n (depth) complex convolutional layers with the parameters passed to it"""
    # Adapt Oren Rippel's paper (https://github.com/oracleofnj/spectral-repr-cnns/tree/master/src/modules)
    
    kl = 'he_uniform'
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(input_tensor)
    else:
        x = input_tensor
            
    for _ in range(n_depth):
        x = convSpectral.ConvComplex2D(rank=2, filters = n_filters, kernel_size = (kernel_size, kernel_size),
                                padding = 'same', kernel_initializer = kl, bias_initializer = kl, kernel_regularizer= 'l2', bias_regularizer= 'l2')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = MonteCarloDropout(dp_rate)(x)
    
    return x


def convSpectral2d_block(input_tensor, n_filters, n_depth=2, input_channel=1, kernel_size = 3, activation = 'relu', dp_rate = 0.1, batchnorm = True):
    """Function to add n (depth) spectral convolutional layers with the parameters passed to it"""
    # Adapt Oren Rippel's paper (https://github.com/oracleofnj/spectral-repr-cnns/tree/master/src/modules)
    
    kl = 'he_uniform'
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(input_tensor)
    else:
        x = input_tensor
            
    for _ in range(n_depth):
        x = convSpectral.ConvSpectral2D(rank=2, filters = n_filters, kernel_size = (kernel_size, kernel_size),
                                padding = 'same', kernel_initializer = kl, bias_initializer = kl, kernel_regularizer= 'l2', bias_regularizer= 'l2')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = MonteCarloDropout(dp_rate)(x)
    
    return x
