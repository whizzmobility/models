""" Hardnet decoder portion, referenced from PingoLH 

Not implemented yet:
New transition layer (after downsampling hardblock)
    Current: 1x1conv, avgpoolx0.5 (densenet transition layer)
    Paper: maxpool input, avgpoolx0.85 hardblock output, concat, 1x1conv
        (0.85x pooling since there is already low-dimension compression, m, within hardnet)
    Supposed outcome - Less CIO at 1x1 conv
"""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.ops import spatial_transform_ops
from official.vision.beta.modeling.layers import nn_blocks

layers = tf.keras.layers

# Each element in 'block' configuration is in the following format:
# (base channels, layers, conv channels after)
HARDNET_SPECS = {
    70: {
      'growth_multiplier': 1.7,
      'downsampling_channels': [16, 24, 32, 48],
      'blocks': [
        (10, 4, 64),
        (16, 4, 96),
        (18, 8, 160),
        (24, 8, 224)
      ]
    }
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class HardNetDecoder(tf.keras.Model):
  """Feature pyramid network."""

  def __init__(self,
               model_id,
               input_specs,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """HardNetDecoder initialization function.

    Args:
      input_specs: `dict` input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
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
        'model_id': model_id,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if use_sync_bn:
      norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      norm = tf.keras.layers.BatchNormalization
    activation_fn = tf.keras.layers.Activation(
        tf_utils.get_activation(activation))

    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    hardnet_specs = HARDNET_SPECS[model_id]
    n_blocks = len(hardnet_specs['blocks'])
    grmul = hardnet_specs['growth_multiplier']

    # last connection is the decoder input, second last is last hardblock output in backbone
    assert len(input_specs) == n_blocks+2, ('Number of input blocks %s ' + \
       'is not equal to expected number of blocks %s') %(len(input_specs), n_blocks+1)
    
    # Build inputs
    inputs = {}
    for level, spec in reversed(input_specs.items()):
      inputs[level] = tf.keras.Input(shape=spec[1:])
    
    skips = dict(inputs)
    x = skips.pop(list(input_specs.keys())[-1]) # decoder input
    skips.pop(list(input_specs.keys())[-2])     # last hardblock output in backbone, processed to decoder input

    for skip, (growth_rate, n_layers, hardblock_channels) in \
     zip(skips.values(), hardnet_specs['blocks'][::-1]):
      
      # Transition up
      x = layers.UpSampling2D(size=2, data_format=data_format, interpolation='bilinear')(x)
      x = tf.concat([x, skip],  axis=bn_axis)
      
      # conv1x1
      x = layers.Conv2D(
        filters=x.shape[bn_axis]//2,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)(x)
      x = norm(
        axis=bn_axis,
        momentum=norm_momentum,
        epsilon=norm_epsilon)(x)
      x = activation_fn(x)

      # denseup
      x = nn_blocks.HardBlock(in_channels=hardblock_channels,
                              growth_rate=growth_rate,
                              growth_multiplier=grmul,
                              n_layers=n_layers)(x)
    
    super(HardNetDecoder, self).__init__(inputs=inputs, outputs={'0': x}, **kwargs)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
