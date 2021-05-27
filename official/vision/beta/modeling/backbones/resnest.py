"""Contains definition of ResNest
Adapted from https://github.com/QiaoranC/tf_ResNeSt_RegNet_model
See https://arxiv.org/pdf/2004.08955.pdf
"""

import tensorflow as tf

from official.modeling import hyperparams
from tensorflow.keras import models
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import (
  Activation,
  Add,
  AveragePooling2D,
  BatchNormalization,
  Conv2D,
  Dense,
  Dropout,
  GlobalAveragePooling2D,
  Input,
  InputSpec,
  MaxPool2D,
  UpSampling2D,
)

from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory

RESNEST_SPECS = {
  50: {
    'blocks_set': [3,4,6,3],
    'stem_width': 32,
  },
  101: {
    'blocks_set': [3,4,23,3],
    'stem_width': 64,
  },
  200: {
    'blocks_set': [3,24,36,3],
    'stem_width': 64,
  },
  269: {
    'blocks_set': [3,30,48,8],
    'stem_width': 64,
  },
}


def get_flops(model):
  run_meta = tf.compat.v1.RunMetadata()
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

  # We use the Keras session graph in the call to the profiler.
  flops = tf.compat.v1.profiler.profile(
    graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd="op", options=opts
  )

  return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(Activation):
  """
  based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
  Mish Activation Function.
  """

  def __init__(self, activation, **kwargs):
    super(Mish, self).__init__(activation, **kwargs)
    self.__name__ = "Mish"


def mish(inputs):
  # with tf.device("CPU:0"):
  result = inputs * tf.math.tanh(tf.math.softplus(inputs))
  return result


class GroupedConv2D(object):
  """Groupped convolution.
  https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_py
  Currently tf.keras and tf.layers don't support group convolution, so here we
  use split/concat to implement this op. It reuses kernel_size for group
  definition, where len(kernel_size) is number of groups. Notably, it allows
  different group has different kernel size.
  """

  def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
    """Initialize the layer.
    Args:
    filters: Integer, the dimensionality of the output space.
    kernel_size: An integer or a list. If it is a single integer, then it is
      same as the original Conv2D. If it is a list, then we split the channels
      and perform different kernel for each group.
    use_keras: An boolean value, whether to use keras layer.
    **kwargs: other parameters passed to the original conv2d layer.
    """
    self._groups = len(kernel_size)
    self._channel_axis = -1

    self._convs = []
    splits = self._split_channels(filters, self._groups)
    for i in range(self._groups):
      self._convs.append(self._get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

  def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
    """A helper function to create Conv2D layer."""
    if use_keras:
      return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
    else:
      return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

  def _split_channels(self, total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    if tf.__version__ < "2.0.0":
      filters = inputs.shape[self._channel_axis].value
    else:
      filters = inputs.shape[self._channel_axis]
    splits = self._split_channels(filters, len(self._convs))
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResNest(tf.keras.Model):
  def __init__(self, 
               model_id,
               input_specs=InputSpec(shape=[None, None, None, 3]),
               stem_type='v1',
               activation="relu",
               dropout_rate=0.2,
               radix=2, 
               groups=1,
               bottleneck_width=64, 
               block_expansion=4, 
               avg_down=True,
               avd=True, 
               avd_first=False, 
               preact=False, 
               using_basic_block=False,
               using_cb=False):
    self.channel_axis = -1  # not for change
    self.model_id = model_id
    self.activation = activation
    self.input_specs = input_specs
    self.dropout_rate = dropout_rate

    self.blocks_set = RESNEST_SPECS[model_id]['blocks_set']
    self.radix = radix
    self.cardinality = groups
    self.bottleneck_width = bottleneck_width

    self.deep_stem = stem_type == 'v1'
    self.stem_width = RESNEST_SPECS[model_id]['stem_width']
    self.block_expansion = block_expansion
    self.avg_down = avg_down
    self.avd = avd
    self.avd_first = avd_first

    self.dilation = 1
    self.preact = preact
    self.using_basic_block = using_basic_block
    self.using_cb = using_cb

    # get_custom_objects().update({'mish': Mish(mish)})

    input_sig = Input(shape=self.input_specs.shape[1:])
    x = self._make_stem(input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

    if self.preact is False:
      x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
      x = tf_utils.get_activation(self.activation)(x)
    print("stem_out", x.shape)

    x = MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_last")(x)
    print("MaxPool2D out", x.shape)

    if self.preact is True:
      x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
      x = tf_utils.get_activation(self.activation)(x)
    
    endpoints = {}
    i = 0
    if self.using_cb:
      second_x = x
      second_x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
      second_x_tmp = self._make_Composite_layer(second_x,filters=x.shape[-1],upsample=False)
      print('layer 0 db_com',second_x_tmp.shape)
      x = Add()([second_x_tmp, x])
    x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
    endpoints[str(i + 2)] = x
    print("-" * 5, "layer 0 out", x.shape, "-" * 5)

    b1_b3_filters = [64,128,256,512]
    for i in range(1, 4):
      if self.using_cb:
        second_x = self._make_layer(x, blocks=self.blocks_set[i], filters=b1_b3_filters[i], stride=2)
        second_x_tmp = self._make_Composite_layer(second_x,filters=x.shape[-1])
        print('layer {} db_com out {}'.format(i,second_x_tmp.shape))
        x = Add()([second_x_tmp, x])
      x = self._make_layer(x, blocks=self.blocks_set[i], filters=b1_b3_filters[i], stride=2)
      print('----- layer {} out {} -----'.format(i,x.shape))
      endpoints[str(i + 2)] = x
    
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    print(self._output_specs)

    super(ResNest, self).__init__(inputs=input_sig, outputs=endpoints)

  def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
    x = input_tensor
    if deep_stem:
      x = Conv2D(stem_width, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
             use_bias=False, data_format="channels_last")(x)

      x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
      x = tf_utils.get_activation(self.activation)(x)

      x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
             kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

      x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
      x = tf_utils.get_activation(self.activation)(x)

      x = Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
             use_bias=False, data_format="channels_last")(x)

      # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
      # x = tf_utils.get_activation(self.activation)(x)
    else:
      x = Conv2D(stem_width, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal",
             use_bias=False, data_format="channels_last")(x)
      # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
      # x = tf_utils.get_activation(self.activation)(x)
    return x

  def _rsoftmax(self, input_tensor, filters, radix, groups):
    x = input_tensor
    batch = x.shape[0]
    if radix > 1:
      x = tf.reshape(x, [-1, groups, radix, filters // groups])
      x = tf.transpose(x, [0, 2, 1, 3])
      x = tf.keras.activations.softmax(x, axis=1)
      x = tf.reshape(x, [-1, 1, 1, radix * filters])
    else:
      x = Activation("sigmoid")(x)
    return x

  def _SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):
    x = input_tensor
    in_channels = input_tensor.shape[-1]

    x = GroupedConv2D(filters=filters * radix, kernel_size=[kernel_size for i in range(groups * radix)],
              use_keras=True, padding="same", kernel_initializer="he_normal", use_bias=False,
              data_format="channels_last", dilation_rate=dilation)(x)

    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    x = tf_utils.get_activation(self.activation)(x)

    batch, rchannel = x.shape[0], x.shape[-1]
    if radix > 1:
      splited = tf.split(x, radix, axis=-1)
      gap = sum(splited)
    else:
      gap = x

    # print('sum',gap.shape)
    gap = GlobalAveragePooling2D(data_format="channels_last")(gap)
    gap = tf.reshape(gap, [-1, 1, 1, filters])
    # print('adaptive_avg_pool2d',gap.shape)

    reduction_factor = 4
    inter_channels = max(in_channels * radix // reduction_factor, 32)

    x = Conv2D(inter_channels, kernel_size=1)(gap)

    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    x = tf_utils.get_activation(self.activation)(x)
    x = Conv2D(filters * radix, kernel_size=1)(x)

    atten = self._rsoftmax(x, filters, radix, groups)

    if radix > 1:
      logits = tf.split(atten, radix, axis=-1)
      out = sum([a * b for a, b in zip(splited, logits)])
    else:
      out = atten * x
    return out

  def _make_block(
    self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
  ):
    x = input_tensor
    inplanes = input_tensor.shape[-1]
    if stride != 1 or inplanes != filters * self.block_expansion:
      short_cut = input_tensor
      if self.avg_down:
        if self.dilation == 1:
          short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
            short_cut
          )
        else:
          short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
        short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same",
                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)
      else:
        short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)

      short_cut = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
    else:
      short_cut = input_tensor
    # should the above be in make layer?
    # see https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnet.py

    group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
    x = Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False,
           data_format="channels_last")(x)
    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    x = tf_utils.get_activation(self.activation)(x)

    avd = avd and (stride > 1 or is_first)

    if avd:
      avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
      stride = 1

    if avd and avd_first:
      x = avd_layer(x)

    if radix >= 1:
      x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                  groups=self.cardinality, radix=radix)
    else:
      x = Conv2D(group_width, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
             dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
      x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
      x = tf_utils.get_activation(self.activation)(x)

    if avd and not avd_first:
      x = avd_layer(x)
      # print('can')
    x = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
           dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

    m2 = Add()([x, short_cut])
    m2 = tf_utils.get_activation(self.activation)(m2)
    return m2

  def _make_block_basic(
    self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
  ):
    """Conv2d_BN_Relu->Bn_Relu_Conv2d
    """
    x = input_tensor
    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    x = tf_utils.get_activation(self.activation)(x)

    short_cut = x
    inplanes = input_tensor.shape[-1]
    if stride != 1 or inplanes != filters * self.block_expansion:
      if self.avg_down:
        if self.dilation == 1:
          short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
            short_cut
          )
        else:
          short_cut = AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
        short_cut = Conv2D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                   use_bias=False, data_format="channels_last")(short_cut)
      else:
        short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal",
                   use_bias=False, data_format="channels_last")(short_cut)

    group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
    avd = avd and (stride > 1 or is_first)
    avd_first = avd_first

    if avd:
      avd_layer = AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
      stride = 1

    if avd and avd_first:
      x = avd_layer(x)

    if radix >= 1:
      x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                  groups=self.cardinality, radix=radix)
    else:
      x = Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
             dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)

    if avd and not avd_first:
      x = avd_layer(x)
      # print('can')

    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    x = tf_utils.get_activation(self.activation)(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
           dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
    m2 = Add()([x, short_cut])
    return m2

  def _make_layer(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
    x = input_tensor
    if self.using_basic_block is True:
      x = self._make_block_basic(x, first_block=True, filters=filters, stride=stride, radix=self.radix,
                     avd=self.avd, avd_first=self.avd_first, is_first=is_first)
      # print('0',x.shape)

      for i in range(1, blocks):
        x = self._make_block_basic(
          x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
        )
        # print(i,x.shape)

    elif self.using_basic_block is False:
      x = self._make_block(x, first_block=True, filters=filters, stride=stride, radix=self.radix, avd=self.avd,
                 avd_first=self.avd_first, is_first=is_first)
      # print('0',x.shape)

      for i in range(1, blocks):
        x = self._make_block(
          x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
        )
        # print(i,x.shape)
    return x

  def _make_Composite_layer(self,input_tensor,filters=256,kernel_size=1,stride=1,upsample=True):
    x = input_tensor
    x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
    x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
    if upsample:
      x = UpSampling2D(size=2)(x)
    return x
  
  def get_config(self):
    return {
      'model_id': self.model_id,
      'blocks_set': self.blocks_set,
      'stem_width': self.stem_width,
      'deep_stem': self.deep_stem,
      'activation': self.activation,
      'dropout_rate': self.dropout_rate,
      
      'radix': self.radix,
      'cardinality': self.cardinality,
      'bottleneck_width': self.bottleneck_width,
      'block_expansion': self.block_expansion,
      'avg_down': self.avg_down,
      'avd': self.avd,
      'avd_first': self.avd_first,

      'dilation': self.dilation,
      'preact': self.preact,
      'using_basic_block': self.using_basic_block,
      'using_cb': self.using_cb
    }
  
  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('resnest')
def build_resnest(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds ResNest backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'resnest', (f'Inconsistent backbone type '
                                     f'{backbone_type}')

  return ResNest(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      stem_type=backbone_cfg.stem_type,
      activation=norm_activation_config.activation)
      # see resnet example to see more configurable options to add
