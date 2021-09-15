import functools

import numpy as np

from tensorflow import int32
from tensorflow import cast
from tensorflow import concat
from tensorflow import constant
from tensorflow import less_equal
from tensorflow import transpose
from tensorflow import complex64
from tensorflow import expand_dims
from tensorflow import math
from tensorflow import dtypes
from tensorflow.signal import fft2d
from tensorflow.signal import rfft2d
from tensorflow.signal import ifft2d
from tensorflow.signal import irfft2d
from tensorflow.signal import fftshift
from tensorflow.signal import ifftshift
from tensorflow.random import uniform
from tensorflow.linalg import normalize

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export

"""
This class (SpectralPool2D) does frequency pooling by converting image into fourier space - works with spectral conv2d (ConvSpectral2D)

Input images should be in real space - this layer does fourier transform of images before the pooling

"""

class SpectralMaxPool2D(Layer):
  """ Spectral pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
  Modified the spectral pool function using tf.fft2d replacing tf.ifft2d in release 0.0.2
  """

  def __init__(self, img_size, 
               alpha = 0.3, 
               beta = 0.15 , 
               gamma = 1.0, 
               kappa = 0.5,
               data_format='channels_last',
               dropout = False,
               name=None, **kwargs):
    super(SpectralMaxPool2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.kappa = kappa
    self.img_size = img_size
    self.dropout = dropout
 
  def call(self, inputs):
    inputs_fft = transpose(inputs, [0,3,1,2])
    im_fft = fft2d(cast(inputs_fft, complex64))
    filter_size = self._get_sp_dim(self.img_size)
    #filter_size = math.multiply(self.gamma, filter_size)
    im_transformed = self._spectral_pool(im_fft, filter_size)
    
    if self.dropout:
        freq_dropout_lower_bound, freq_dropout_upper_bound = self._get_frq_dropout_bounds(filter_size, self.kappa)
        if (freq_dropout_lower_bound is not None and
            freq_dropout_upper_bound is not None):
            tf_random_cutoff = uniform([], freq_dropout_lower_bound, freq_dropout_upper_bound)
            dropout_mask = self._frequency_dropout_mask(filter_size,
                                                        tf_random_cutoff)
        
            im_downsampled = im_transformed * dropout_mask
        else:
            im_downsampled = im_transformed
    else:
        im_downsampled = im_transformed
    
    outputs = transpose(math.real(ifft2d(im_downsampled)), [0,2,3,1])
    
    return outputs
    
    
  def _spectral_pool(self, images, filter_size):
    assert len(images.get_shape().as_list()) == 4
    assert filter_size >= 3

    if filter_size % 2 == 1:
        n = int((filter_size-1) // 2)
        top_left = images[:, :, :n+1, :n+1]
        top_right = images[:, :, :n+1, -n:]
        bottom_left = images[:, :, -n:, :n+1]
        bottom_right = images[:, :, -n:, -n:]
        top_combined = concat([top_left, top_right], axis=-1)
        bottom_combined = concat([bottom_left, bottom_right], axis=-1)
        all_together = concat([top_combined, bottom_combined], axis=-2)
    else:
        n = int(filter_size // 2)
        top_left = images[:, :, :n, :n]
        top_middle = expand_dims(
            cast(0.5 ** 1.0, complex64) *
            (images[:, :, :n, n] + images[:, :, :n, -n]),
            -1)
        top_right = images[:, :, :n, -(n-1):]
        middle_left = expand_dims(
            cast(0.5 ** 1.0, complex64) *
            (images[:, :, n, :n] + images[:, :, -n, :n]), 
            -2)
        middle_middle = expand_dims(
            expand_dims(
                cast(0.5, complex64) *
                (images[:, :, n, n] + images[:, :, n, -n] +
                images[:, :, -n, n] + images[:, :, -n, -n]),
                -1), -1)
        middle_right = expand_dims(
            cast(0.5 ** 1.0, complex64) *
            (images[:, :, n, -(n-1):] + images[:, :, -n, -(n-1):]),
            -2)
        bottom_left = images[:, :, -(n-1):, :n]
        bottom_middle = expand_dims(
            cast(0.5 ** 1.0, complex64) *
            (images[:, :, -(n-1):, n] + images[:, :, -(n-1):, -n]),
            -1)
        bottom_right = images[:, :, -(n-1):, -(n-1):]
        top_combined = concat(
            [top_left, top_middle, top_right],
            axis=-1)
        middle_combined = concat(
            [middle_left, middle_middle, middle_right],
            axis=-1)
        bottom_combined = concat(
            [bottom_left, bottom_middle, bottom_right],
            axis=-1)
        all_together = concat(
            [top_combined, middle_combined, bottom_combined],
            axis=-2)
    return all_together
    
  def _get_frq_dropout_bounds(self, n, m):
    c = self.alpha + m* (self.beta - self.alpha)
    freq_dropout_lower_bound = c * (1. + n // 2)
    freq_dropout_upper_bound = (1. + n // 2)

    return freq_dropout_lower_bound, freq_dropout_upper_bound

  def _get_sp_dim(self, n):
        if n % 2 == 1:
            fsize = int((n-1) // 2)
        else:
            fsize = int(n // 2)
        # minimum size is 3:
        return max(3, fsize)

  def _frequency_dropout_mask(self, height, frequency_to_truncate_above):
    cutoff_shape = frequency_to_truncate_above.get_shape().as_list()
    assert len(cutoff_shape) == 0

    mid = int(height // 2)
    if height % 2 == 1:
        go_to = mid + 1
    else:
        go_to = mid
    indexes = np.concatenate((
        np.arange(go_to),
        np.arange(mid, 0, -1)
    )).astype(np.float32)

    xs = np.broadcast_to(indexes, (height, height))
    ys = np.broadcast_to(np.expand_dims(indexes, -1), (height, height))
    highest_frequency = np.maximum(xs, ys)
    print(frequency_to_truncate_above)

    comparison_mask = constant(highest_frequency)
    dropout_mask = cast(less_equal(
        comparison_mask,
        frequency_to_truncate_above
    ), complex64)
    
    return dropout_mask

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def get_config(self):
    config = {
        'img_size': self.img_size,
        'alpha': self.alpha,
        'beta': self.beta,
        'gamma': self.gamma,
        'kappa': self.kappa,
        'data_format': self.data_format,
        'name':self.name
    }
    base_config = super(SpectralMaxPool2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



"""
This class (Complex SpectralPool2D) does frequency (Spectral pooling) pooling of the images already in fourier space - works with ConvComplex2D

The input to this pooling layer is a complex fourier image (augmented real and imaginary values such as dimension of batch of images - [batch_size, H, W, input_channel*2] where the input_channel*2 are for real and imag part of the fft of image)

"""

class ComplexSpectralMaxPool2D(Layer):
  """ Spectral pooling layer for arbitrary pooling functions, for 2D complex inputs (e.g. fourier images).
  Modified the spectral pool function using tf.fft2d replacing tf.ifft2d in release 0.0.2
  """

  def __init__(self, img_size, 
               filters , 
               alpha = 0.3, 
               beta = 0.15 , 
               gamma = 1.0, 
               kappa = 0.5,
               data_format='channels_last', 
               name=None, **kwargs):
    super(ComplexSpectralMaxPool2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)
    self.img_size = img_size
    self.filters = filters
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.kappa = kappa
 
  def call(self, inputs):
    im_fft = dtypes.complex(inputs[:,:,:,:self.filters], inputs[:,:,:,self.filters:])
    im_fft = transpose(im_fft, [0,3,1,2])
    
    filter_size = self._get_sp_dim(self.img_size)
    #filter_size = math.multiply(self.gamma, filter_size)
    im_transformed = self._spectral_pool(im_fft, filter_size)
    
    freq_dropout_lower_bound, freq_dropout_upper_bound = self._get_frq_dropout_bounds(filter_size, self.kappa)
    if (freq_dropout_lower_bound is not None and
        freq_dropout_upper_bound is not None):
        
        tf_random_cutoff = uniform([], freq_dropout_lower_bound, freq_dropout_upper_bound)
        dropout_mask = self._frequency_dropout_mask(
                            filter_size,
                            tf_random_cutoff)
        
        im_downsampled = im_transformed * dropout_mask
    else:
        im_downsampled = im_transformed
    
    outputs_downsampled = transpose(im_downsampled, [0,2,3,1])
    outputs = concat([math.real(outputs_downsampled), math.imag(outputs_downsampled)], axis = -1)
    
    return outputs
    
    
  def _spectral_pool(self, images, filter_size):
    assert len(images.get_shape().as_list()) == 4
    assert filter_size >= 3

    if filter_size % 2 == 1:
        n = int((filter_size-1)/2)
        top_left = images[:, :, :n+1, :n+1]
        top_right = images[:, :, :n+1, -n:]
        bottom_left = images[:, :, -n:, :n+1]
        bottom_right = images[:, :, -n:, -n:]
        top_combined = concat([top_left, top_right], axis=-1)
        bottom_combined = concat([bottom_left, bottom_right], axis=-1)
        all_together = concat([top_combined, bottom_combined], axis=-2)
    else:
        n = filter_size // 2
        top_left = images[:, :, :n, :n]
        top_middle = expand_dims(
            cast(0.5 ** 0.5, complex64) *
            (images[:, :, :n, n] + images[:, :, :n, -n]),
            -1)
        top_right = images[:, :, :n, -(n-1):]
        middle_left = expand_dims(
            cast(0.5 ** 0.5, complex64) *
            (images[:, :, n, :n] + images[:, :, -n, :n]), 
            -2)
        middle_middle = expand_dims(
            expand_dims(
                cast(0.5, complex64) *
                (images[:, :, n, n] + images[:, :, n, -n] +
                images[:, :, -n, n] + images[:, :, -n, -n]),
                -1), -1)
        middle_right = expand_dims(
            cast(0.5 ** 0.5, complex64) *
            (images[:, :, n, -(n-1):] + images[:, :, -n, -(n-1):]),
            -2)
        bottom_left = images[:, :, -(n-1):, :n]
        bottom_middle = expand_dims(
            cast(0.5 ** 0.5, complex64) *
            (images[:, :, -(n-1):, n] + images[:, :, -(n-1):, -n]),
            -1)
        bottom_right = images[:, :, -(n-1):, -(n-1):]
        top_combined = concat(
            [top_left, top_middle, top_right],
            axis=-1)
        middle_combined = concat(
            [middle_left, middle_middle, middle_right],
            axis=-1)
        bottom_combined = concat(
            [bottom_left, bottom_middle, bottom_right],
            axis=-1)
        all_together = concat(
            [top_combined, middle_combined, bottom_combined],
            axis=-2)
    return all_together
    
  def _get_frq_dropout_bounds(self, n, m):
    c = self.alpha + m* (self.beta - self.alpha)
    freq_dropout_lower_bound = c * (1. + n // 2)
    freq_dropout_upper_bound = (1. + n // 2)

    return freq_dropout_lower_bound, freq_dropout_upper_bound

  def _get_sp_dim(self, n):
        fsize = int(self.gamma * n)
        # minimum size is 3:
        return max(3, fsize)

  def _frequency_dropout_mask(self, height, frequency_to_truncate_above):
    cutoff_shape = frequency_to_truncate_above.get_shape().as_list()
    assert len(cutoff_shape) == 0

    mid = height // 2
    if height % 2 == 1:
        go_to = mid + 1
    else:
        go_to = mid
    indexes = np.concatenate((
        np.arange(go_to),
        np.arange(mid, 0, -1)
    )).astype(np.float32)

    xs = np.broadcast_to(indexes, (height, height))
    ys = np.broadcast_to(np.expand_dims(indexes, -1), (height, height))
    highest_frequency = np.maximum(xs, ys)

    comparison_mask = constant(highest_frequency)
    dropout_mask = cast(less_equal(
        comparison_mask,
        frequency_to_truncate_above
    ), complex64)
    
    return dropout_mask

  def get_config(self):
    config = {
        'img_size': self.img_size,
        'filters':self.filters,
        'alpha': self.alpha,
        'beta': self.beta,
        'gamma': self.gamma,
        'kappa': self.kappa,
        'data_format': self.data_format,
        'name': self.name
    }
    base_config = super(ComplexSpectralMaxPool2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
