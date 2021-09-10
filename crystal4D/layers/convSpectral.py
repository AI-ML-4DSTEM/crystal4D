import functools
import six

from tensorflow import cast
from tensorflow import Variable
from tensorflow import math
from tensorflow import dtypes
from tensorflow import transpose
from tensorflow.random import uniform
from tensorflow import zeros_like
from tensorflow import concat

from tensorflow.signal import fft2d
from tensorflow.signal import rfft2d
from tensorflow.signal import ifft2d
from tensorflow.signal import irfft2d
from tensorflow.signal import fftshift
from tensorflow.signal import ifftshift

from tensorflow.keras.backend import permute_dimensions

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
# pylint: disable=g-classes-have-attributes

"""
This class (ConvComplex2D) implements the complex convolution between two complex vectors (W = A + iB) and input vector (X = x + iy). The weights initialized in this class are complex and can be convolved with complex inputs such as fourier representation of image inputs...

Adapted from implementation of complex convolutions in Deep Complex Networks by Trabelsi et. al. (2018)

#ToDo: will use complex batch normalization
"""

class ConvComplex2D(Layer):
  """2-D complex convolution layer
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    super(ConvComplex2D, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
        filters = int(filters)
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

  
  #kernel for spectral CNN
  def kernel_initialize(self, fan_in, fan_out):
    limit = math.sqrt(6/fan_in)
    return uniform(minval = -limit, maxval = limit, shape = [fan_in, fan_out, self.kernel_size[0], self.kernel_size[1]])
        
  def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
        raise ValueError(
            'The number of filters must be evenly divisible by the number of '
            'groups. Received: groups={}, filters={}'.format(
                self.groups, self.filters))

    if not all(self.kernel_size):
        raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                        'Received: %s' % (self.kernel_size,))

    if (self.padding == 'causal' and not isinstance(self, (Conv1D, SeparableConv1D))):
        raise ValueError('Causal padding is only supported for `Conv1D`'
                        'and `SeparableConv1D`.')

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    self.input_channel = self._get_input_channel(input_shape)
    if self.input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, self.input_channel,
                                                    input_shape))
    
    
    self.kernel_shape = self.kernel_size  + (self.input_channel // 2 , self.filters*2)

    #fourier kernel (spectral)
    
    self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
    
    if self.use_bias:
        bias_shape = (self.filters*2,)
        self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
    
        #self.bias = dtypes.complex(bias_fft, zeros_like(bias_fft), name='bias')
    else:
        self.bias = None
        
    #############################
    
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: self.input_channel})

    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
        tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, six.string_types):
        tf_padding = self.padding.upper()
    else:
        tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    
    tf_op_name = self.__class__.__name__
    '''
    if tf_op_name == 'Conv1D':
        tf_op_name = 'conv1d'  # Backwards compat.
    '''

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)
    self.built = True

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
    
    #Complex convolution
    kernel_real = self.kernel[:,:,:,:self.filters]
    
    kernel_imag = self.kernel[:,:,:,self.filters:]
    
    cat_kernels_4_real = concat([kernel_real, -kernel_imag], axis=-2)
    cat_kernels_4_imag = concat([kernel_imag, kernel_real], axis=-2)
    cat_kernels_4_complex = concat(
            [cat_kernels_4_real, cat_kernels_4_imag], axis=-1
        )
    
    outputs = self._convolution_op(inputs, cat_kernels_4_complex)
    #print(outputs.shape)
    ###############
    
    if self.use_bias:
        output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
            bias = array_ops.reshape(self.bias, (1, self.filters, 1))
            outputs += bias
        else:
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                def _apply_fn(o):
                    return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = nn.bias_add(
                    outputs, self.bias, data_format=self._tf_data_format)

    ################
    
    if self.activation is not None:
        return self.activation(outputs)

    return outputs


  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            input_shape[:batch_rank]
            + self._spatial_output_shape(input_shape[batch_rank:-1])
            + [self.filters])
    else:
        return tensor_shape.TensorShape(
            input_shape[:batch_rank] + [self.filters] +
            self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    return False

  def get_config(self):
    config = {
        'filters':
             self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
             self.data_format,
        'dilation_rate':
             self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(ConvComplex2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
         batch_rank = 1
    else:
         batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
        causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
        causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
        return -1 - self.rank
    else:
        return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
        raise ValueError('The channel dimension of the inputs '
                      'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
        op_padding = 'valid'
    else:
        op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
        op_padding = op_padding.upper()
    return op_padding




"""
This class (ConvSpectral2D) implements the spectral parametrization for learning weights in fourier space. In order to work around the complexity of complex convolution, the class levarage (Rippel et. al.) conjugate symmetry (irfft2d) for real valued signals...

#ToDo: 
"""

class ConvSpectral2D(Layer):
  """2-D spectral convolution layer
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    super(ConvSpectral2D, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
        filters = int(filters)
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

  
  #kernel for spectral CNN
  def kernel_initialize(self, fan_in, fan_out):
    limit = math.sqrt(6/fan_in)
    return uniform(minval = -limit, maxval = limit, shape = [fan_in, fan_out, self.kernel_size[0], self.kernel_size[1]])
        
  def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
        raise ValueError(
            'The number of filters must be evenly divisible by the number of '
            'groups. Received: groups={}, filters={}'.format(
                self.groups, self.filters))

    if not all(self.kernel_size):
        raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                        'Received: %s' % (self.kernel_size,))

    if (self.padding == 'causal' and not isinstance(self, (Conv1D, SeparableConv1D))):
        raise ValueError('Causal padding is only supported for `Conv1D`'
                        'and `SeparableConv1D`.')

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    self.input_channel = self._get_input_channel(input_shape)
    if self.input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, self.input_channel,
                                                    input_shape))
    
    
    self.kernel_shape = self.kernel_size  + (self.input_channel , self.filters*2)

    #fourier kernel (spectral)
    
    #print(self.kernel_shape)
    
    self.kernel_spatial = self.add_weight(
            name="kernel",
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
    
    if self.use_bias:
        bias_shape = (self.filters,)
        self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
    
        #self.bias = dtypes.complex(bias_fft, zeros_like(bias_fft), name='bias')
    else:
        self.bias = None
        
    #############################
    
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: self.input_channel})

    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
        tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, six.string_types):
        tf_padding = self.padding.upper()
    else:
        tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    tf_op_name = self.__class__.__name__
    if tf_op_name == 'Conv1D':
        tf_op_name = 'conv1d'  # Backwards compat.

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)
    self.built = True

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
    
    #Spectral CNN 
    
    kernel_spatial_trans = cast(transpose(self.kernel_spatial, [2, 3, 0, 1]), dtype=dtypes.float32)
    
    kernel = math.real(ifft2d(dtypes.complex(kernel_spatial_trans[:,:self.filters,:,:], kernel_spatial_trans[:,self.filters:,:,:])))
    
    kernel = cast(transpose(kernel, [2, 3, 0, 1]), dtype=self.dtype)
    
    outputs = self._convolution_op(inputs, kernel)
    #print(outputs.shape)
    ###############
    
    if self.use_bias:
        output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
            bias = array_ops.reshape(self.bias, (1, self.filters, 1))
            outputs += bias
        else:
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                def _apply_fn(o):
                    return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = nn.bias_add(
                    outputs, self.bias, data_format=self._tf_data_format)

    ################
    
    if self.activation is not None:
        return self.activation(outputs)

    return outputs


  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            input_shape[:batch_rank]
            + self._spatial_output_shape(input_shape[batch_rank:-1])
            + [self.filters])
    else:
        return tensor_shape.TensorShape(
            input_shape[:batch_rank] + [self.filters] +
            self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    return False

  def get_config(self):
    config = {
        'filters':
             self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
             self.data_format,
        'dilation_rate':
             self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(ConvSpectral2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
         batch_rank = 1
    else:
         batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
        causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
        causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
        return -1 - self.rank
    else:
        return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
        raise ValueError('The channel dimension of the inputs '
                      'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
        op_padding = 'valid'
    else:
        op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
        op_padding = op_padding.upper()
    return op_padding