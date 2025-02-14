# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains definitions of instance prediction heads."""

from typing import List, Union, Optional, Mapping
# Import libraries
import tensorflow as tf

from official.modeling import tf_utils

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectionHead(tf.keras.layers.Layer):
  """Creates a detection head."""

  def __init__(
      self,
      num_classes: int,
      num_convs: int = 0,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      num_fcs: int = 2,
      fc_dims: int = 1024,
      class_agnostic_bbox_pred: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a detection head.

    Args:
      num_classes: An `int` for the number of classes.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the FC layers.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      num_fcs: An `int` number that represents the number of FC layers before
        the predictions.
      fc_dims: An `int` number that represents the number of dimension of the FC
        layers.
      class_agnostic_bbox_pred: `bool`, indicating whether bboxes should be
        predicted for every class or not.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(DetectionHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'num_fcs': num_fcs,
        'fc_dims': fc_dims,
        'class_agnostic_bbox_pred': class_agnostic_bbox_pred,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
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
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'detection-conv_{}'.format(i)
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'detection-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._fcs = []
    self._fc_norms = []
    for i in range(self._config_dict['num_fcs']):
      fc_name = 'detection-fc_{}'.format(i)
      self._fcs.append(
          tf.keras.layers.Dense(
              units=self._config_dict['fc_dims'],
              kernel_initializer=tf.keras.initializers.VarianceScaling(
                  scale=1 / 3.0, mode='fan_out', distribution='uniform'),
              kernel_regularizer=self._config_dict['kernel_regularizer'],
              bias_regularizer=self._config_dict['bias_regularizer'],
              name=fc_name))
      bn_name = 'detection-fc-bn_{}'.format(i)
      self._fc_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._classifier = tf.keras.layers.Dense(
        units=self._config_dict['num_classes'],
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='detection-scores')

    num_box_outputs = (4 if self._config_dict['class_agnostic_bbox_pred'] else
                       self._config_dict['num_classes'] * 4)
    self._box_regressor = tf.keras.layers.Dense(
        units=num_box_outputs,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='detection-boxes')

    super(DetectionHead, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool = None):
    """Forward pass of box and class branches for the Mask-RCNN model.

    Args:
      inputs: A `tf.Tensor` of the shape [batch_size, num_instances, roi_height,
        roi_width, roi_channels], representing the ROI features.
      training: a `bool` indicating whether it is in `training` mode.

    Returns:
      class_outputs: A `tf.Tensor` of the shape
        [batch_size, num_rois, num_classes], representing the class predictions.
      box_outputs: A `tf.Tensor` of the shape
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions.
    """
    roi_features = inputs
    _, num_rois, height, width, filters = roi_features.get_shape().as_list()

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation(x)

    _, _, _, filters = x.get_shape().as_list()
    x = tf.reshape(x, [-1, num_rois, height * width * filters])

    for fc, bn in zip(self._fcs, self._fc_norms):
      x = fc(x)
      x = bn(x)
      x = self._activation(x)

    classes = self._classifier(x)
    boxes = self._box_regressor(x)
    return classes, boxes

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class MaskHead(tf.keras.layers.Layer):
  """Creates a mask head."""

  def __init__(
      self,
      num_classes: int,
      upsample_factor: int = 2,
      num_convs: int = 4,
      num_filters: int = 256,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      class_agnostic: bool = False,
      **kwargs):
    """Initializes a mask head.

    Args:
      num_classes: An `int` of the number of classes.
      upsample_factor: An `int` that indicates the upsample factor to generate
        the final predicted masks. It should be >= 1.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the mask prediction layers.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      class_agnostic: A `bool`. If set, we use a single channel mask head that
        is shared between all classes.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(MaskHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'upsample_factor': upsample_factor,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'class_agnostic': class_agnostic
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'mask-conv_{}'.format(i)
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'mask-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._deconv = tf.keras.layers.Conv2DTranspose(
        filters=self._config_dict['num_filters'],
        kernel_size=self._config_dict['upsample_factor'],
        strides=self._config_dict['upsample_factor'],
        padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='mask-upsampling')
    self._deconv_bn = bn_op(name='mask-deconv-bn', **bn_kwargs)

    if self._config_dict['class_agnostic']:
      num_filters = 1
    else:
      num_filters = self._config_dict['num_classes']

    conv_kwargs = {
        'filters': num_filters,
        'kernel_size': 1,
        'padding': 'valid',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    self._mask_regressor = conv_op(name='mask-logits', **conv_kwargs)

    super(MaskHead, self).build(input_shape)

  def call(self, inputs: List[tf.Tensor], training: bool = None):
    """Forward pass of mask branch for the Mask-RCNN model.

    Args:
      inputs: A `list` of two tensors where
        inputs[0]: A `tf.Tensor` of shape [batch_size, num_instances,
          roi_height, roi_width, roi_channels], representing the ROI features.
        inputs[1]: A `tf.Tensor` of shape [batch_size, num_instances],
          representing the classes of the ROIs.
      training: A `bool` indicating whether it is in `training` mode.

    Returns:
      mask_outputs: A `tf.Tensor` of shape
        [batch_size, num_instances, roi_height * upsample_factor,
         roi_width * upsample_factor], representing the mask predictions.
    """
    roi_features, roi_classes = inputs
    batch_size, num_rois, height, width, filters = (
        roi_features.get_shape().as_list())
    if batch_size is None:
      batch_size = tf.shape(roi_features)[0]

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation(x)

    x = self._deconv(x)
    x = self._deconv_bn(x)
    x = self._activation(x)

    logits = self._mask_regressor(x)

    mask_height = height * self._config_dict['upsample_factor']
    mask_width = width * self._config_dict['upsample_factor']

    if self._config_dict['class_agnostic']:
      logits = tf.reshape(logits, [-1, num_rois, mask_height, mask_width, 1])
    else:
      logits = tf.reshape(
          logits,
          [-1, num_rois, mask_height, mask_width,
           self._config_dict['num_classes']])

    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_rois])
    mask_indices = tf.tile(
        tf.expand_dims(tf.range(num_rois), axis=0), [batch_size, 1])

    if self._config_dict['class_agnostic']:
      class_gather_indices = tf.zeros_like(roi_classes, dtype=tf.int32)
    else:
      class_gather_indices = tf.cast(roi_classes, dtype=tf.int32)

    gather_indices = tf.stack(
        [batch_indices, mask_indices, class_gather_indices],
        axis=2)
    mask_outputs = tf.gather_nd(
        tf.transpose(logits, [0, 1, 4, 2, 3]), gather_indices)
    return mask_outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class YOLOv3Head(tf.keras.layers.Layer):
  """Creates a YOLOv3Head head."""

  def __init__(
      self,
      levels: int,
      num_classes: int,
      strides: List,
      anchor_per_scale: int,
      anchors: List,
      xy_scale: List,
      kernel_initializer='VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a YOLOv3Head head.
    Referenced from YOLOv4 implementation on:
    https://github.com/hunglc007/tensorflow-yolov4-tflite

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      levels: An `int` number of feature scales from decoder
      strides: A `List` with `int` denoting stride for bbox location prediction
      anchor_per_scale: `int`, number of anchors per scale
      anchors: A `List` with `int` denoting width and height scaling for bbox
      xy_scale: A `List` with `int` denoting x, y scaling for bbox location prediction
        Length of list should correspond to number of branches.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(YOLOv3Head, self).__init__(**kwargs)

    self._config_dict = {
        'levels': levels,
        'num_classes': num_classes,
        'strides': strides,
        'anchor_per_scale': anchor_per_scale,
        'anchors': anchors,
        'xy_scale': xy_scale,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

    self.levels = levels
    self.anchor_per_scale = anchor_per_scale
    self.anchors = tf.constant(anchors, dtype=tf.float32)
    self.anchors = tf.reshape(self.anchors, [
      int(len(anchors)/self.anchor_per_scale/2), self.anchor_per_scale, 2])
    self.num_classes = num_classes
    self.strides = strides
    self.xy_scale = xy_scale
  
  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the classification head."""
    self.heads = {}
    for i in range(self.levels):
      self.heads[str(i)] = layers.Conv2D(
        filters=self.anchor_per_scale * (self.num_classes + 5),
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._config_dict['kernel_initializer'],
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    super(YOLOv3Head, self).build(input_shape)

  def call(self, 
           backbone_output: Mapping[str, tf.Tensor],
           decoder_output: Mapping[str, tf.Tensor]):
    """Forward pass of the YOLOv3 head.

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

    outputs = {'raw_outputs': {}, 'predictions': {}}

    for i, branch in enumerate(decoder_output.values()):
      x = branch
      x = self.heads[str(i)](x)
      outputs['raw_outputs'][str(i)] = x

      x_shape = x.shape
      x = tf.stack(tf.split(x, 3, axis=-1), axis=-2) # [b, h, w, x] -> [b, h, w, 3, x/3]

      raw_dxdy, raw_dwdh, raw_conf, raw_prob = tf.split(x, (2, 2, 1, self.num_classes),
                                                        axis=self._bn_axis)

      xy_grid = tf.meshgrid(tf.range(x_shape[1]), tf.range(x_shape[2]))
      xy_grid = tf.stack(xy_grid, axis=self._bn_axis) # [gx, gy, 2]
      xy_grid = tf.expand_dims(tf.expand_dims(xy_grid, axis=2), axis=0) # [1, gx, gy, 1, 2]
      xy_grid = tf.cast(xy_grid, tf.float32)

      pred_xy = ((tf.sigmoid(raw_dxdy) * self.xy_scale[i]) - 0.5 * \
                (self.xy_scale[i] - 1) + xy_grid) * self.strides[i]
      pred_wh = (tf.exp(raw_dwdh) * self.anchors[i])
      pred_xywh = tf.concat([pred_xy, pred_wh], axis=self._bn_axis)

      pred_conf = tf.sigmoid(raw_conf)
      pred_prob = tf.sigmoid(raw_prob)

      outputs['predictions'][str(i)] = tf.concat(
        [pred_xywh, pred_conf, pred_prob], axis=self._bn_axis)

    return outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
