import sys
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

#Custom libraries - User defined
from ..utils import conv_utils
from ..layers import poolSpectral
from ..utils.data_utils import parseDataset
from ..utils.conv_utils import convSpectral2d_block, conv2d_block
from ..utils.conv_utils import cross_correlate_iff, cross_correlate_ff, cross_correlate

def spectralUnet(input_channel, image_size, filter_size, n_depth, dp_rate = 0.1, act = 'relu'):
    
    input_size = (image_size, image_size, input_channel)
    inputsA = tf.keras.Input(shape=input_size)   # input for CBED
    inputsB = tf.keras.Input(shape=input_size)   # input for probe
    skips = []

    print("Cross correlation layer ... \n")
    cc1 = tf.keras.layers.Lambda(cross_correlate)([inputsA, inputsB])
    #cc1 = tf.keras.layers.BatchNormalization()(cc1)
    
    print("\n")
    print("Building the U-net ... \n")
    
    pool = convSpectral2d_block(cc1, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    
    # conv u-net for 4DSTEM cbed pattern
    for encode in range(int(np.log2(image_size))-1):
        conv = convSpectral2d_block(pool, np.minimum(filter_size * (2**encode), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv)
        skips.append(conv)
    
    for decode in reversed(range(int(np.log2(256))-1)):
        conv = convSpectral2d_block(pool, np.minimum(filter_size * (2**(decode+1)), 256), n_depth=n_depth, dp_rate=dp_rate, activation=act)
        pool = tf.keras.layers.UpSampling2D(size=(2, 2))(conv)
        pool = tf.keras.layers.Concatenate(axis=-1)([skips[decode], pool])

    unet_final = convSpectral2d_block(pool, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act) 
    
    print("\n")
    print("Building the one more convolution layers ... \n")
    conv = conv2d_block(unet_final, filter_size, n_depth=n_depth, dp_rate=dp_rate, activation=act)

    print("\n")
    print("Building the final layer ... \n")
    pred_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3),padding='same', kernel_initializer = 'he_uniform')(conv)
    pred_out = tf.keras.layers.Activation('linear', dtype='float32')(pred_out)

    model = tf.keras.Model(inputs=[inputsA, inputsB], outputs=pred_out)

    return model