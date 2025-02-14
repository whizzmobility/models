""" Referenced from YOLOv4 implementation on: 
https://github.com/hunglc007/tensorflow-yolov4-tflite

Path Aggregation Net: https://arxiv.org/abs/1803.01534
"""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils

layers = tf.keras.layers

PANET_SPECS = {
  # number of PA routes: multiplier to num_filters for each PA step
  2: [1.0, 1.0],
  3: [1.0, 0.5, 1.0, 2.0]
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class PAN(tf.keras.Model):
  """Decoder for Path-Aggregation Network feature in YOLOv4"""

  def __init__(self,
               input_specs,
               routes,
               num_filters=256,
               num_convs=5,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """PAN initialization function.

    Args:
      input_specs: `dict` input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      routes: number of path aggregation routes
      num_filters: `int` base number of filters in PAN convolutions.
      num_convs: `int` number of convs for each PAN step
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
        'num_filters': num_filters,
        'num_convs': num_convs,
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
    self.activation_fn = tf_utils.get_activation(activation)

    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_last':
      self.bn_axis = -1
    else:
      self.bn_axis = 1
    
    if num_convs % 2 == 0:
      raise ValueError('num_convs in PAN should be an odd number, got %s' %num_convs)
    
    self.norm_momentum = norm_momentum
    self.norm_epsilon = norm_epsilon
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer

    # Build inputs
    inputs = {}
    feature_size = None
    for level, spec in reversed(input_specs.items()):
      if feature_size == spec[1] and spec[1] is not None:
        continue
      inputs[level] = tf.keras.Input(shape=spec[1:])
      feature_size = spec[1]

    routeIdx = [i for i in list(inputs.keys())][:routes]
    skips = []
    outputs = {}
    deep_route = inputs[routeIdx[0]]

    # aggregate decreasing depth, store skips
    for i in range(routes-1):
      skips.append(deep_route)
      shallow_route = inputs[routeIdx[i+1]]
      filters = int(PANET_SPECS[routes][i] * num_filters)

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      shallow_route = self.conv(shallow_route, filters=filters, kernels=1)

      deep_route = layers.UpSampling2D(size=2, data_format=data_format)(deep_route)
      deep_route = tf.concat([deep_route, shallow_route], axis=self.bn_axis)

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      for _ in range(num_convs//2):
        deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
        deep_route = self.conv(deep_route, filters=filters, kernels=1)
    
    # aggregate increasing depth, pop skips, store outputs
    for i in range(routes-1):
      filters = int(PANET_SPECS[routes][i + routes - 1] * num_filters)

      outputs[str(i)] = self.conv(deep_route, filters=filters, kernels=3)
      deep_route = self.conv(deep_route, filters=filters, kernels=3, downsample=True)
      deep_route = tf.concat([deep_route, skips.pop()], axis=self.bn_axis)

      deep_route = self.conv(deep_route, filters=filters, kernels=1)
      for _ in range(num_convs//2):
        deep_route = self.conv(deep_route, filters=filters*2, kernels=3)
        deep_route = self.conv(deep_route, filters=filters, kernels=1)

    outputs[str(routes-1)] = self.conv(deep_route, filters=filters*2, kernels=3)
    self._output_specs = {k: v.get_shape() for k, v in outputs.items()}

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

    return self.activation_fn(x)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
