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

# Lint as: python3
"""Tests for instance_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.heads import instance_heads
from official.vision.beta.modeling.decoders import pan


class DetectionHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (0, 0, False, False),
      (0, 1, False, False),
      (1, 0, False, False),
      (1, 1, False, False),
  )
  def test_forward(self, num_convs, num_fcs, use_separable_conv, use_sync_bn):
    detection_head = instance_heads.DetectionHead(
        num_classes=3,
        num_convs=num_convs,
        num_filters=16,
        use_separable_conv=use_separable_conv,
        num_fcs=num_fcs,
        fc_dims=4,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    roi_features = np.random.rand(2, 10, 128, 128, 16)
    scores, boxes = detection_head(roi_features)
    self.assertAllEqual(scores.numpy().shape, [2, 10, 3])
    self.assertAllEqual(boxes.numpy().shape, [2, 10, 12])

  def test_serialize_deserialize(self):
    detection_head = instance_heads.DetectionHead(
        num_classes=91,
        num_convs=0,
        num_filters=256,
        use_separable_conv=False,
        num_fcs=2,
        fc_dims=1024,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = detection_head.get_config()
    new_detection_head = instance_heads.DetectionHead.from_config(config)
    self.assertAllEqual(
        detection_head.get_config(), new_detection_head.get_config())


class MaskHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (1, 1, False),
      (1, 2, False),
      (2, 1, False),
      (2, 2, False),
  )
  def test_forward(self, upsample_factor, num_convs, use_sync_bn):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        upsample_factor=upsample_factor,
        num_convs=num_convs,
        num_filters=16,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    roi_features = np.random.rand(2, 10, 14, 14, 16)
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    self.assertAllEqual(
        masks.numpy().shape,
        [2, 10, 14 * upsample_factor, 14 * upsample_factor])

  def test_serialize_deserialize(self):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        upsample_factor=2,
        num_convs=1,
        num_filters=256,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = mask_head.get_config()
    new_mask_head = instance_heads.MaskHead.from_config(config)
    self.assertAllEqual(
        mask_head.get_config(), new_mask_head.get_config())

  def test_forward_class_agnostic(self):
    mask_head = instance_heads.MaskHead(
        num_classes=3,
        class_agnostic=True
    )
    roi_features = np.random.rand(2, 10, 14, 14, 16)
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    self.assertAllEqual(masks.numpy().shape, [2, 10, 28, 28])


class YOLOv3HeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (
        3, 3,
        256,
        [8, 16, 32], 
        [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401], \
        [1.2, 1.1, 1.05]
      ), # yolo
      (
        2, 3,
        256,
        [16, 32], 
        [23,27, 37,58, 81,82, 81,82, 135,169, 344,319], \
        [1.05, 1.05]
      ) # yolotiny
  )
  def test_forward(self, levels, anchor_per_scale, input_size, strides, anchors, xy_scale):
    yolov3_head = instance_heads.YOLOv3Head(
        levels=levels,
        num_classes=80,
        strides=strides,
        anchor_per_scale=anchor_per_scale,
        anchors=anchors,
        xy_scale=xy_scale,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None
    )

    channels = pan.PANET_SPECS[levels][0]
    size = input_size // 2**(6 - levels + 1) # hardnet downsamples 6x
    decoder_features = {i: np.random.rand(1, int(size*0.5**i), int(size*0.5**i), channels*2**i)
      for i in range(levels)}
    
    pred = yolov3_head({}, decoder_features)
    for i in range(0, len(pred), 2):
      self.assertAllEqual(
        [1, size, size, 3*(80+5)],
        pred['raw_outputs'][str(i)].shape.as_list()
      )
      self.assertAllEqual(
        [1, size, size, 3, 80+5],
        pred['predictions'][str(i)].shape.as_list()
      )
      size /= 2

  def test_serialize_deserialize(self):
    yolov3_head = instance_heads.YOLOv3Head(
        levels=3,
        num_classes=80,
        strides=[16, 32],
        anchor_per_scale=3,
        anchors=[23,27, 37,58, 81,82, 81,82, 135,169, 344,319],
        xy_scale=[1.05, 1.05],
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None
    )
    config = yolov3_head.get_config()
    new_yolov3_head = instance_heads.YOLOv3Head.from_config(config)
    self.assertAllEqual(
        yolov3_head.get_config(), new_yolov3_head.get_config())


if __name__ == '__main__':
  tf.test.main()
