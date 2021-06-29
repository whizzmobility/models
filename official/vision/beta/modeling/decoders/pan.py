""" Referenced from YOLOv4 implementation on: 
https://github.com/hunglc007/tensorflow-yolov4-tflite

Path Aggregation Net: https://arxiv.org/abs/1803.01534
"""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils

layers = tf.keras.layers

PANET_SPECS = {
  3: [256, 128, 256, 512]
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class PAN(tf.keras.Model):
  """Decoder for Path-Aggregation Network feature in YOLOv4"""

  def __init__(self,
               input_specs,
               routes,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """PAN initialization function.
    Takes last three levels to perform PAN for YOLOv4.
    TODO: extend it to support specifying layers

    Args:
      input_specs: `dict` input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      routes: number of path aggregation routes
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'routes': routes,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if use_sync_bn:
      self.norm = layers.experimental.SyncBatchNormalization
    else:
      self.norm = layers.BatchNormalization
    activation_fn = layers.Activation(
        tf_utils.get_activation(activation))

    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_last':
      self.bn_axis = -1
    else:
      self.bn_axis = 1
    
    self.norm_momentum = norm_momentum
    self.norm_epsilon = norm_epsilon
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer

    # Build inputs
    inputs = {}
    input_specs.pop(list(input_specs.keys())[-2])
    for level, spec in reversed(input_specs.items()):
      inputs[level] = tf.keras.Input(shape=spec[1:])
    
    routeIdx = [i for i in list(reversed(input_specs.keys()))][:routes]
    skips = []
    outputs = {}
    deep_route = inputs[routeIdx[0]]

    # aggregate decreasing depth, store skips
    for i in range(routes-1):
      skips.append(deep_route)
      shallow_route = inputs[routeIdx[i+1]]
      filters = PANET_SPECS[routes][i]

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      shallow_route = self.conv(shallow_route, filters=filters, kernels=1)

      deep_route = layers.UpSampling2D(size=2, data_format=data_format)(deep_route)
      deep_route = tf.concat([deep_route, shallow_route], axis=self.bn_axis)

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=1)
    
    # aggregate increasing depth, pop skips, store outputs
    for i in range(routes-1):
      filters = PANET_SPECS[routes][i + routes - 1]

      outputs[i] = self.conv(deep_route, filters=filters, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=3, downsample=True)
      deep_route = tf.concat([deep_route, skips.pop()], axis=self.bn_axis)

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=1)

    outputs[routes-1] = self.conv(deep_route, filters=filters*2, kernels=3)

    super(PAN, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

  def conv(self,
           inputs: tf.Tensor,
           filters: int,
           kernels: int,
           strides: int = 1,
           downsample: bool = False):
    """Creates one group of conv-bn-activation block.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      filters: An `int` number of filters for the first convolution of the
        layer.
      kernels: An `int` number representing size of kernel.
      strides: An `int` stride to use for the first convolution of the layer.
        If greater than 1, this layer will downsample the input.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    padding = 'same'
    if downsample:
      inputs = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
      padding = 'valid'
      strides = 2

    x = layers.Conv2D(
      filters=filters,
      kernel_size=kernels,
      strides=strides,
      padding=padding,
      use_bias=False,
      kernel_initializer=self.kernel_initializer,
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer)(inputs)

    x = self.norm(
      axis=self.bn_axis,
      momentum=self.norm_momentum,
      epsilon=self.norm_epsilon)(x)

    return layers.Activation(
        tf_utils.get_activation('relu'))(x)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
