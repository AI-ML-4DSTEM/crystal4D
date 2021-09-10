import numpy as np

import functools
import six

from tensorflow import Variable
from tensorflow import math
from tensorflow import dtypes
from tensorflow import transpose
from tensorflow.random import uniform
from tensorflow import zeros_like
from tensorflow import ones_like
from tensorflow import concat

from tensorflow import int32
from tensorflow import float32
from tensorflow import cast
from tensorflow import concat
from tensorflow import constant
from tensorflow import less_equal
from tensorflow import transpose
from tensorflow import complex64
from tensorflow import expand_dims
from tensorflow import math
from tensorflow import dtypes
from tensorflow import divide
from tensorflow import multiply
from tensorflow.keras.initializers import random_uniform
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.util.tf_export import keras_export

from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

class multiTaskLayer(Layer):
    def __init__(self, sigma_size = 2,
                sigma_initializer = None,
                scale_factor = 0.1,
                sigma_penalty = 0.0001):
        super(multiTaskLayer, self).__init__()
        self.sigma_size = sigma_size
        self.scale_factor = scale_factor
        self.sigma_shape = [self.sigma_size,]
        self.sigma_penalty = sigma_penalty
        self._scale_factor = constant([scale_factor,0])
        if sigma_initializer:
            self.sigma_initializer = sigma_initializer
        else:
            self.sigma_initializer = random_uniform(minval=0.25, maxval=1.0)
            
                                                 
        self._sigma_sq = self.add_weight(
            name="sigma_sq",
            shape=self.sigma_shape,
            initializer=self.sigma_initializer,
            constraint=constraints.non_neg(),
            trainable=True,
            dtype=self.dtype)

    def call(self, inputs):
        assert(isinstance(inputs, list) or isinstance(inputs, tuple)), 'inputs to the layer should be either list or tuple of tensors'
        true_input = inputs[0]
        pred_input = inputs[1]
        
        assert(len(true_input.shape) == len(pred_input.shape)), 'true and prediction should have same dimension!!'
        true_input = cast(true_input, dtype=self._sigma_sq.dtype)
        pred_input = cast(pred_input, dtype=self._sigma_sq.dtype)
        
        #Multiply Vg to subsequent tasks (such as loss term for qz = square(true-pred)*true_vg)
        true_vg = expand_dims(true_input[:,:,:,0], axis=-1)
        padding = ones_like(true_vg, dtype=self._sigma_sq.dtype)
        true_vg = concat([padding, true_vg], axis=-1)
        
        self._sigma_sq_factor = multiply(cast(self._scale_factor, dtype=float32) 
                                         , divide(1.0 , multiply(2.0, (cast(self._sigma_sq, dtype=float32 )+ self.sigma_penalty))))
        
        self._log_sigma_sq = math.log(cast(self._sigma_sq, dtype=float32 )+ self.sigma_penalty)
        loss_tensor = multiply(self._sigma_sq_factor, math.square(pred_input-true_input)*true_vg) + self._log_sigma_sq
        losses = math.reduce_sum(loss_tensor, axis=-1)
        outputs = cast(pred_input, self.dtype)
        
        return outputs, losses
    
    def get_config(self):
        config = {
            'sigma_size':
            self.sigma_size,
            'sigma_initializer':
            initializers.serialize(self.sigma_initializer),
            'scale_factor':
            self.scale_factor,
            'sigma_penalty':
            self.sigma_penalty
        }
        base_config = super(multiTaskLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
