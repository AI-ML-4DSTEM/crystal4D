#Added through version 0.0.6 release
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

def spectralPoolDenseNet(input_channel, image_size, filter_size, dense_size, n_depth, n_conv, n_dense, dp_rate = 0.1, act = 'relu'):
    #Added through version 0.0.6 release
    input_size = (image_size, image_size, input_channel)
    inputs = tf.keras.Input(shape=input_size)

    pool = inputs

    print("\n")
    print("Building the spectral Dense-net blocks ... \n")
    for block in range(n_conv):
        if image_size[0]//(2**block) > 8:
            conv = convSpectral2d_block(pool,
                                        filter_size * (2**block),
                                        n_depth= n_depth,
                                        dp_rate= dp_rate,
                                        activation= act)
            pool = poolSpectral.SpectralMaxPool2D(img_size = image_size[0]//(2**block))(conv)

    print("\n")
    print("Building the some fully connected layers ... \n")
    dense = tf.keras.layers.Flatten()(pool)
    
    for dense_layer in reversed(range(n_dense)):
        dense = tf.keras.layers.Dense(dense_layer_size * (2**dense_layer),
                                     activation= act,
                                     kernel_regularizer='l2')(dense)
        dense = tf.keras.layers.Dropout(dp_rate)(dense)

    print("\n")
    print("Building the final output dense layer ... \n")
    output = tf.keras.layers.Dense(1,
                                   activation='softmax',
                                   kernel_regularizer='l2',
                                   dtype='float32')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model