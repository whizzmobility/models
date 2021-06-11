"""Contains definitions of classification heads."""
from typing import List, Union, Optional, Mapping
import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class ClassificationHead(tf.keras.layers.Layer):
  """Creates a classification head."""

  def __init__(
      self,
      num_classes: int,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      add_head_batch_norm: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      dropout_rate: float = 0.0,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a classification head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      level: An `int` or `str`, level to use to build classification head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      add_head_batch_norm: `bool` whether to add a batch normalization layer
        before pool.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      dropout_rate: `float` rate for dropout regularization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ClassificationHead, self).__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'add_head_batch_norm': add_head_batch_norm,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'dropout_rate': dropout_rate,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the classification head."""
    conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
        'kernel_size': 3,
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.01),
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'classification_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              **conv_kwargs))
      norm_name = 'classification_head_norm_{}'.format(i)
      self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    self._head_norm = None
    if self._config_dict['add_head_batch_norm']:
      self._head_norm = bn_op(name='classification_head_initial_norm', **bn_kwargs)
    self._dropout = tf.keras.layers.Dropout(self._config_dict['dropout_rate'])
    self._classifier = tf.keras.layers.Dense(
        name='classification_output',
        units=self._config_dict['num_classes'],
        kernel_initializer='random_uniform',
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    super(ClassificationHead, self).build(input_shape)

  def call(self, 
           backbone_output: Mapping[str, tf.Tensor],
           decoder_output: Mapping[str, tf.Tensor]):
    """Forward pass of the classification head.

    Args:
      backbone_output: A `dict` of tensors, to utilise SegmentationModel's 
        implementation
      decoder_output: A `dict` of tensors
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].
    Returns:
      classification prediction: A `tf.Tensor` of the classification 
        scores predicted from input features.
    """
    x = decoder_output[str(self._config_dict['level'])]
    
    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation(x)

    if self._head_norm is not None:
      x = self._head_norm(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = self._dropout(x)
    return self._classifier(x)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
