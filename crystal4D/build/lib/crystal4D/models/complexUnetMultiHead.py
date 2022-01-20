import sys
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

#Custom libraries - User defined
from ..utils import conv_utils
from ..layers import poolSpectral, multiTaskLayer
from ..utils.data_utils import parseDataset
from ..utils.conv_utils import convSpec2d_block, conv2d_block
from ..utils.conv_utils import cross_correlate_iff, cross_correlate_ff, cross_correlate

def complexUnetMultiHead(input_channel, out_channel, image_size, filter_size, n_depth, dp_rate = 0.1, act = 'relu'):
    
    input_size = (image_size, image_size, input_channel // 2)
    inputsA = tf.keras.Input(shape=input_size)   # input for CBED
    inputsB = tf.keras.Input(shape=input_size)   # input for probe
    
    output_size = (image_size, image_size, out_channel)
    true_out = tf.keras.Input(shape=output_size)   # input for probe

    skips = []

    print("Cross correlation layer ... \n")
    cc1 = tf.keras.layers.Lambda(cross_correlate_ff)([inputsA, inputsB])
    #cc1 = tf.keras.layers.BatchNormalization()(cc1)
    
    print("\n")
    print("Building the comlpex U-net ... \n")
    
    pool = convSpec2d_block(cc1, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    
    # conv u-net for 4DSTEM cbed pattern
    for encode in range(int(np.log2(image_size))-1):
        conv = convSpec2d_block(pool, np.minimum(filter_size * (2**encode), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv)
        skips.append(conv)
        
    poolVg = pool
    poolQz= pool
    
    for decode in reversed(range(int(np.log2(image_size))-1)):
        convVg = convSpec2d_block(poolVg, np.minimum(filter_size * (2**(decode+1)), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        poolVg = tf.keras.layers.UpSampling2D(size=(2, 2))(convVg)
        poolVg = tf.keras.layers.Concatenate(axis=-1)([skips[decode], poolVg])
        
    for decode in reversed(range(int(np.log2(image_size))-1)):
        convQz = convSpec2d_block(poolQz, np.minimum(filter_size * (2**(decode+1)), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        poolQz = tf.keras.layers.UpSampling2D(size=(2, 2))(convQz)
        poolQz = tf.keras.layers.Concatenate(axis=-1)([skips[decode], poolQz])

    unetVg = convSpec2d_block(poolVg, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    unetQz = convSpec2d_block(poolQz, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 

    print("\n")
    print("Building the inverse fft layer ... \n")
    unet_real_vg = tf.keras.layers.Lambda(cross_correlate_iff, dtype='float32')(unetVg)
    unet_real_qz = tf.keras.layers.Lambda(cross_correlate_iff, dtype='float32')(unetQz)
    
    print("\n")
    print("Building the one more spatial convolution layers ... \n")
    conv_real_vg = conv2d_block(unet_real_vg, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act)
    conv_real_qz = conv2d_block(unet_real_qz, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act)

    print("\n")
    print("Building the final layer ... \n")
    pot_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3),padding='same', kernel_initializer = 'he_uniform')(conv_real_vg)
    qz_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3),padding='same', kernel_initializer = 'he_uniform')(conv_real_qz)
    
    pred_out = tf.keras.layers.Concatenate(axis=-1)([pot_out, qz_out])
    pred_out, losses = multiTaskLayer(scale_factor=0.001)([true_out, pred_out])
    pred_out = tf.keras.layers.Activation('linear', dtype='float32')(pred_out)
    
    model = tf.keras.Model(inputs=[inputsA, inputsB, true_out], outputs=(pred_out, losses))

    return model