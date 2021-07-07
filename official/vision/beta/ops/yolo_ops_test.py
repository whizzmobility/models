"""Tests for yolo_ops.py."""

import io
# Import libraries
from absl.testing import parameterized
import numpy as np

import tensorflow as tf

from official.vision.beta.ops import yolo_ops
from official.vision.beta.projects.yolo.ops import box_ops


class YoloOpsTest(parameterized.TestCase, tf.test.TestCase):
  
  @parameterized.parameters(
    (
      [0.456250, 0.415104, 0.176563, 0.171875], # xywh in normalised form
      [84.26663, 94.19994, 128.26663, 139.40007], # yxyx in pixels
      960, 1280, (256, 256), False
    ),
    (
      [0.464453, 0.517188, 0.249219, 0.065625], # xywh in normalised form
      [253.00009, 86.99993, 265.6001, 150.8], # yxyx in pixels
      960, 1280, (256, 256), True
    )
  )
  def testResizeImageBoxes(self, 
                           bbox, 
                           bbox_result, 
                           height, 
                           width, 
                           target_dim, 
                           preserve_aspect_ratio):
    image = tf.random.uniform((height, width, 3))
    bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    bbox = box_ops.xcycwh_to_yxyx(bbox)
    new_image, new_bbox = yolo_ops.resize_image_and_bboxes(
      image=image, 
      bboxes=bbox, 
      target_size=target_dim, 
      preserve_aspect_ratio=preserve_aspect_ratio)
    
    self.assertAllClose(new_bbox, bbox_result)
    self.assertAllEqual(new_image.shape[:2], target_dim)

  def testYoloPreprocessTrueBoxes(self):
    bboxes = tf.constant([
        [ 40,  79, 109, 144,  74],
        [174, 242, 187, 269,  24],
        [341, 265, 357, 291,  26],
        [261, 220, 300, 362,   0],
        [217, 228, 252, 338,   0],
        [202, 228, 219, 274,   0],
        [135, 232, 153, 278,   0],
        [ 94, 229, 124, 306,   0],
        [ 44, 232,  74, 321,   0],
        [191, 238, 196, 255,  24],
        [117,  86, 122, 157,  74],
        [180, 224, 193, 285,   0],
        [375, 226, 415, 326,   0],
        [245, 222, 274, 317,   0],
        [317, 228, 352, 334,   0],
        [369, 226, 389, 263,   0],
        [135, 225, 180, 355,   0],
        [171, 229, 185, 311,   0],
        [  0, 216, 415, 363,   0]]) # x1, y1, x2, y2, class
    
    classes = bboxes[:, 4]
    bboxes = tf.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3], bboxes[:, 2]], axis=-1) #yxyx
    bboxes = box_ops.yxyx_to_xcycwh(tf.cast(bboxes, tf.float32))
    inputs = tf.concat([bboxes, tf.cast(classes[:, tf.newaxis], tf.float32)], axis=-1)
    
    train_output_sizes=tf.constant([52,26,13])
    anchor_per_scale = 3
    num_classes = 80
    max_bbox_per_scale = 150
    strides = tf.constant([8, 16, 32])
    anchors = tf.constant([[[ 12,  16], [ 19,  36], [ 40,  28]],[[ 36,  75], [ 76,  55], [ 72, 146]],[[142, 110], [192, 243], [459, 401]]])

    result = yolo_ops.preprocess_true_boxes(
      bboxes=inputs,
      train_output_sizes=train_output_sizes,
      anchor_per_scale=anchor_per_scale,
      num_classes=num_classes,
      max_bbox_per_scale=max_bbox_per_scale,
      strides=strides,
      anchors=anchors) # only takes xywh

    target_labels, target_bboxes = result

    groundtruth_label_small_bbox = np.array([ 
        74.5, 111.5,  69. ,  65. ,   1. ,   1. , 119.5, 121.5,   5. ,
        71. ,   1. ,   1. , 193.5, 246.5,   5. ,  17. ,   1. ,   1. ,
       379. , 244.5,  20. ,  37. ,   1. ,   1. , 144. , 255. ,  18. ,
        46. ,   1. ,   1. , 180.5, 255.5,  13. ,  27. ,   1. ,   1. ,
       186.5, 254.5,  13. ,  61. ,   1. ,   1. , 210.5, 251. ,  17. ,
        46. ,   1. ,   1. , 109. , 267.5,  30. ,  77. ,   1. ,   1. ,
       178. , 270. ,  14. ,  82. ,   1. ,   1. , 259.5, 269.5,  29. ,
        95. ,   1. ,   1. ,  59. , 276.5,  30. ,  89. ,   1. ,   1. ,
       349. , 278. ,  16. ,  26. ,   1. ,   1. , 395. , 276. ,  40. ,
       100. ,   1. ,   1. , 234.5, 283. ,  35. , 110. ,   1. ,   1. ,
       334.5, 281. ,  35. , 106. ,   1. ,   1. , 157.5, 290. ,  45. ,
       130. ,   1. ,   1. , 207.5, 289.5, 415. , 147. ,   1. ,   1. ,
       280.5, 291. ,  39. , 142. ,   1. ,   1. ])
    groundtruth_small_bbox = np.array([
        74.5, 111.5,  69. ,  65. , 180.5, 255.5,  13. ,  27. , 349. ,
       278. ,  16. ,  26. , 280.5, 291. ,  39. , 142. , 234.5, 283. ,
        35. , 110. , 210.5, 251. ,  17. ,  46. , 144. , 255. ,  18. ,
        46. , 109. , 267.5,  30. ,  77. ,  59. , 276.5,  30. ,  89. ,
       193.5, 246.5,   5. ,  17. , 119.5, 121.5,   5. ,  71. , 186.5,
       254.5,  13. ,  61. , 395. , 276. ,  40. , 100. , 259.5, 269.5,
        29. ,  95. , 334.5, 281. ,  35. , 106. , 379. , 244.5,  20. ,
        37. , 157.5, 290. ,  45. , 130. , 178. , 270. ,  14. ,  82. ,
       207.5, 289.5, 415. , 147. ])
    
    self.assertAllClose(
      tf.boolean_mask(target_labels[0], tf.greater(target_labels[0], 0.5)), 
      groundtruth_label_small_bbox)
    self.assertAllClose(
      tf.boolean_mask(target_bboxes[0], tf.greater(target_bboxes[0], 0.5)),
      groundtruth_small_bbox)
    self.assertAllEqual(target_labels[0].shape, np.array([52, 52, 3, 85]))
    self.assertAllEqual(target_bboxes[0].shape, np.array([150, 4]))
    self.assertAllEqual(target_labels[1], tf.zeros([26, 26, 3, 85]))
    self.assertAllEqual(target_bboxes[1], tf.zeros([150, 4]))
    self.assertAllEqual(target_labels[2], tf.zeros([13, 13, 3, 85]))
    self.assertAllEqual(target_bboxes[2], tf.zeros([150, 4]))


if __name__ == '__main__':
  tf.test.main()
