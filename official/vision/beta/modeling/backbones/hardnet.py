""" Contains definition of FCHardnet, referenced from PingoLH 

Implementation differs from paper:
Layers and growth rates:
    Paper hardnet-68: 
        layers: [8,(16,16),16,4]
        growth rate: [14, 16, 20, 40, 160]
        (emphasis on stride-8 for local feature learning)
    Implementation hardnet-70:
        layers: [4, 4, 8, 8, 8]
        growth rate: [10, 16, 18, 24, 32]
    Hardnet-70 is a segmentation task whereas models in paper are used for classification
    Can't seem to find any papers supporting this implementation

PingoLH hardblock-v2 (not pulled) includes conv with biases
"""

import logging
# Import libraries
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
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
        (24, 8, 224),
        (32, 8, 320)
      ]
    }
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class HardNet(tf.keras.Model):
  """Class to build Hardnet family model."""

  def __init__(self,
               model_id,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Hardnet initialization function.

    Args:
      model_id: `int` depth of Hardnet backbone model.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      **kwargs: keyword arguments to be passed.
    """
    self._model_id = model_id
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1
    
    hardnet_specs = HARDNET_SPECS[model_id]
    n_blocks = len(hardnet_specs['blocks'])

    # Build HardNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    x = inputs

    # Downsampling
    for i, channels in enumerate(hardnet_specs['downsampling_channels']):
      strides = 1
      if i in [0, 2]:
        strides = 2
      
      x = layers.Conv2D(
        filters=channels,
        kernel_size=3,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(x)
      x = self._norm(
        axis=bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)(x)
      x = tf_utils.get_activation(activation)(x)

    # HardBlocks
    endpoints = {}
    grmul = hardnet_specs['growth_multiplier']
    
    for i, (growth_rate, n_layers, conv_channels) in enumerate(hardnet_specs['blocks']):
      blk = nn_blocks.HardBlock(in_channels=channels,
                                growth_rate=growth_rate,
                                growth_multiplier=grmul,
                                n_layers=n_layers)
      x = blk(x)
      endpoints[str(i)] = x # skip connections comes after hardblock, before conv2d + avgpool

      # Densenet transition layer! Not the mapping method shown in paper that reduces CIO @ conv
      x = layers.Conv2D(
        filters=conv_channels,
        kernel_size=3,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(x)
      channels = conv_channels

      if i < n_blocks - 1:
        x = layers.AveragePooling2D(pool_size=(2, 2), 
                                    strides=2)(x)
    endpoints[str(i+1)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    for i, v in endpoints.items():
      logging.info(f"{i}, {v}, {v.shape}")

    super(HardNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('hardnet')
def build_hardnet(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds Hardnet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'hardnet', (f'Inconsistent backbone type '
                                     f'{backbone_type}')

  return HardNet(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
