import functools
import six

from tensorflow import reshape
from tensorflow import einsum
from tensorflow import convert_to_tensor

from tensorflow.math import reduce_sum

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
from tensorflow.image import extract_patches


class ConvQuad2D(Layer):
    def __init__(self,
                   rank,
                   filters,
                   kernel_size,
                   strides=(1,1),
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
            super(ConvQuad2D, self).__init__(
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                **kwargs)
            self.rank = 2

            if isinstance(filters, float):
                filters = int(filters)
            self.filters = filters
            self.groups = groups or 1
            self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
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
    
    
            #Quadtraic CNN initialization
    
            if isinstance(kernel_size, int):
                self.quad_kernel_size = conv_utils.normalize_tuple(
                    (kernel_size**2), rank, 'quad_kernel_size')
            else:
                self.quad_kernel_size = conv_utils.normalize_tuple(
                    (kernel_size[0]**2,kernel_size[1]**2), rank, 'quad_kernel_size')
    
    #Quadratic CNN using Volterra kernel theory
    def _volterra_conv(self, inputs, W, input_dim, quad_ksize, quad_strides, padding):
        input_patches = extract_patches(inputs,
                                        sizes= quad_ksize,
                                        strides=quad_strides,
                                        rates=quad_strides,
                                        padding=padding)
       
        input_patches_shape = (-1, 250, 250, self.kernel_size[0]*self.kernel_size[1], input_dim)
        #print(input_patches_shape)
        input_patches = array_ops.reshape(input_patches, input_patches_shape)
        V = einsum('abcid,abcjd,dijo->abcdo', input_patches, input_patches, W)
        return reduce_sum(V, 3)
    ##############################################

    def _validate_init(self):
            if self.filters is not None and self.filters % self.groups != 0:
                raise ValueError(
                      'The number of filters must be evenly divisible by the number of '
                      'groups. Received: groups={}, filters={}'.format(
                          self.groups, self.filters))

            if not all(self.kernel_size):
                raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                                'Received: %s' % (self.kernel_size,))

            if (self.padding == 'causal' and not isinstance(self,(Conv1D, SeparableConv1D))):
                  raise ValueError('Causal padding is only supported for `Conv1D`'
                                   'and `SeparableConv1D`.')


    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                  'The number of input channels must be evenly divisible by the number '
                  'of groups. Received groups={}, but the input has {} channels '
                  '(full input shape is {}).'.format(self.groups, input_channel,
                                                     input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
    
        #Volterra kernel initialize
    
        self.quad_kernel_shape = (input_channel // self.groups,) + self.quad_kernel_size + (self.filters,)
    
        self.quad_kernel = self.add_weight(
            name='quad_kernel',
            shape=self.quad_kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
    
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

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
    
        #Volterra quad CNN
        tf_quad_strides = list((1,) + self.strides + (1,))
        tf_quad_padding = tf_padding
        tf_quad_ksize = list((1,) +  self.kernel_size + (1,))
    
        self._quad_convolution_op = functools.partial(
            self._volterra_conv,
            input_dim = input_channel,
            quad_ksize = tf_quad_ksize,
            quad_strides=tf_quad_strides,
            padding=tf_quad_padding)
    
        self.built = True

    def call(self, inputs):
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self._quad_convolution_op(inputs, self.quad_kernel) + self._convolution_op(inputs, self.kernel)

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

                outputs = nn_ops.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format=self._tf_data_format)

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
            'quad_kernel_size':
                self.quad_kernel_size,
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
        base_config = super(ConvQuad2D, self).get_config()
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


