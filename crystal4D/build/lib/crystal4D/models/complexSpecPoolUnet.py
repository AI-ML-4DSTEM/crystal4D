import sys
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

#Custom libraries - User defined
from ..utils import conv_utils
from ..layers import poolSpectral
from ..utils.data_utils import parseDataset
from ..utils.conv_utils import convSpec2d_block, conv2d_block
from ..utils.conv_utils import cross_correlate_iff, cross_correlate_ff, cross_correlate

def complexSpecPoolUnet(input_channel, image_size, filter_size, n_depth, dp_rate = 0.1, act = 'relu'):
    
    input_size = (image_size, image_size, input_channel)
    inputsA = tf.keras.Input(shape=input_size)   # input for CBED
    inputsB = tf.keras.Input(shape=input_size)   # input for probe
    skips = []

    print("Cross correlation layer ... \n")
    cc1 = tf.keras.layers.Lambda(cross_correlate_ff)([inputsA, inputsB])
    #cc1 = tf.keras.layers.BatchNormalization()(cc1)
    
    print("\n")
    print("Building the comlpex U-net ... \n")
    
    pool = convSpec2d_block(cc1, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    
    # conv u-net for 4DSTEM cbed pattern
    for encode in range(int(np.log2(image_size))-1):
        if image_size//(2**encode) > 8:
            conv = convSpec2d_block(pool, np.minimum(filter_size * (2**encode), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
            pool = poolSpectral.ComplexSpectralMaxPool2D(image_size//(2**encode), np.minimum(filter_size * (2**encode), 256))(conv)
            skips.append(conv)
    
    for decode in reversed(range(len(skips))):
        conv = convSpec2d_block(pool, np.minimum(filter_size * (2**(decode+1)), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
        pool = tf.keras.layers.Concatenate(axis=-1)([skips[decode], pool])

    unet_final = convSpec2d_block(pool, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 

    print("\n")
    print("Building the inverse fft layer ... \n")
    unet_out = tf.keras.layers.Lambda(cross_correlate_iff, dtype='float32')(unet_final)
    
    print("\n")
    print("Building the one more spatial convolution layers ... \n")
    conv_real = conv2d_block(unet_out, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act)

    print("\n")
    print("Building the final layer ... \n")
    pred_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3),padding='same', kernel_initializer = 'he_uniform')(conv_real)
    pred_out = tf.keras.layers.Activation('linear', dtype='float32')(pred_out)
    
    model = tf.keras.Model(inputs=[inputsA, inputsB], outputs=pred_out)

    return model