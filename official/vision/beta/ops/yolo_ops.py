"""Utility ops for yolo data.
Referenced from https://github.com/hunglc007/tensorflow-yolov4-tflite.
"""

from typing import List

import numpy as np
import tensorflow as tf


def resize_image_and_bboxes(image: tf.Tensor, 
                            bboxes: tf.Tensor, 
                            target_size: Tuple[int, int],
                            preserve_aspect_ratio: bool = False,
                            image_height: int = None,
                            image_width: int = None,
                            image_normalized: bool = True):
  """
  Args:
    image: `tf.Tensor` of shape (None, 5), denoting (x, y, w, h, class), non-normalized
    bboxes: `tf.Tensor` of shape (None, 4), denoting (ymin, xmin, ymax, xmax), non-normalized
    target: `Tuple[int,int]`, denoting height and width of resulting image/bbox
    preserve_aspect_ratio: `bool`, true to preserve image aspect ratio
    image_height: `int`, height of image
    image_width: `int`, width of image
  
  !! assumes image is normalized to 0-1
  """
  target_height, target_width = target_size
  if image_height is None or image_width is None:
    image_height, image_width, _ = image.shape
  scale_height, scale_width = target_height / image_height, target_width / image_width

  if preserve_aspect_ratio:
    clip_size = max(image_height, image_width)
    pad_height = (clip_size - image_height)//2
    pad_width = (clip_size - image_width)//2

    if image_normalized:
      image = tf.pad(image, 
        tf.constant([[pad_height, pad_height], [pad_width, pad_width], [0, 0]]), 
        constant_values=0.5)
    else:
      image = tf.image.pad_to_bounding_box(
        image, pad_height, pad_width, clip_size, clip_size)

    scale = min(scale_height, scale_width)
    bboxes *= scale
    offset = tf.stack([pad_height, pad_width, pad_height, pad_width], axis=-1)
    bboxes += tf.cast(offset, tf.float32)
  
  else:
    scale = tf.stack([scale_height, scale_width, scale_height, scale_width], axis=-1)
    bboxes *= tf.cast(scale, tf.float32)

  image = tf.image.resize(image, target_size)

  return image, bboxes


def preprocess_true_boxes(bboxes: tf.Tensor,
                          train_output_sizes: List[int],
                          anchor_per_scale: int,
                          num_classes: int,
                          max_bbox_per_scale: int,
                          strides: List[int],
                          anchors: tf.Tensor,
                          is_bbox_in_pixels=True,
                          is_xywh=False):
  """
  train_output_sizes: `List[int]`, dimension of each scaled feature map
  anchor_per_scale: `int`, number of anchors per scale
  num_classes: `int`, number of classes.
  max_bbox_per_Scale: `int`, maximum number of bounding boxes per scale.
  strides: `List[int]` of output strides, ratio of input to output resolution.
  anchors: `tf.Tensor` of shape (None, anchor_per_scale, 2) denothing positions
    of anchors
  is_bbox_in_pixels: `bool`, true if bounding box values are in pixels
  is_xywh: `bool`, true if bounding box values are in (x, y, width, height) format

  !!! Assumes the images and boxes are preprocessed to fit image size.
  Scaling will be according to output sizes predicted by output strides
  """
  
  if (is_bbox_in_pixels and is_xywh):
    raise NotImplementedError('Processing for xywh in pixel format not implemented.')
  if (not is_bbox_in_pixels and is_bbox_in_pixels):
    raise NotImplementedError('Processing for box corners in float format not implemented')

  max_output_size = tf.reduce_max(train_output_sizes)
  label = tf.zeros((len(strides), max_output_size, max_output_size, anchor_per_scale, 5+num_classes))

  bboxes_xywh = tf.zeros((len(strides), max_bbox_per_scale, 4))
  bbox_count = tf.zeros((3,))
  const = tf.constant([1.0], dtype=tf.float32)

  for bbox in bboxes:
    if is_bbox_in_pixels:
      bbox_coor = tf.cast(bbox[:4], tf.float32)
      bbox_xywh = tf.concat([
          (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
          bbox_coor[2:] - bbox_coor[:2],
        ], axis=-1)
    else:
      bbox_xywh = tf.cast(bbox[:4], tf.float32)

    bbox_class_ind = tf.cast(bbox[4], tf.int64)

    smooth_onehot = tf.one_hot(
      bbox_class_ind, num_classes, off_value=1.0/num_classes)

    bbox_xywh_scaled = tf.repeat(
      bbox_xywh[tf.newaxis, :], repeats=len(strides), axis=0)
    bbox_xywh_scaled /= [[i] for i in strides]

    bbox_label = tf.concat([bbox_xywh, const, smooth_onehot], axis=-1)

    iou = []
    exist_positive = False
    # register for each anchor setting
    for i in range(3):
      anchors_xywh = tf.add(tf.floor(bbox_xywh_scaled[i, 0:2]), 0.5)
      anchors_xywh = tf.repeat(anchors_xywh[tf.newaxis, :], 3, axis=0) 
      anchors_xywh = tf.concat([anchors_xywh, tf.cast(anchors[i], tf.float32)], axis=-1)

      iou_scale = bbox_iou(
        bbox_xywh_scaled[i][tf.newaxis, :], anchors_xywh
      )
      iou.append(iou_scale)
      iou_mask = iou_scale > 0.3

      if tf.reduce_any(iou_mask):
        xind = tf.cast(tf.floor(bbox_xywh_scaled[i, 0]), tf.int32)
        yind = tf.cast(tf.floor(bbox_xywh_scaled[i, 1]), tf.int32)

        update = tf.gather([tf.zeros_like(bbox_label), bbox_label], tf.cast(iou_mask, tf.int64))
        label = tf.tensor_scatter_nd_update(label, indices=[[i, yind, xind]], updates=[update])

        bbox_ind = tf.cast(bbox_count[i] % max_bbox_per_scale, tf.int32)
        bboxes_xywh = tf.tensor_scatter_nd_update(bboxes_xywh, indices=[[i, bbox_ind]], updates=[bbox_xywh])
        bbox_count = tf.tensor_scatter_nd_add(bbox_count, indices=[[i]], updates=[1])

        exist_positive = True

    # registers for best anchor
    if not exist_positive:
      best_anchor_ind = tf.argmax(tf.concat(iou, axis=-1), axis=-1)
      best_detect = tf.cast(best_anchor_ind / anchor_per_scale, tf.int32)
      best_anchor = tf.cast(best_anchor_ind % anchor_per_scale, tf.int32)
      
      xind = tf.cast(tf.floor(bbox_xywh_scaled[best_detect, 0]), tf.int32)
      yind = tf.cast(tf.floor(bbox_xywh_scaled[best_detect, 1]), tf.int32)

      label = tf.tensor_scatter_nd_update(
        label, indices=[[best_detect, yind, xind, best_anchor]], updates=[bbox_label])

      bbox_ind = tf.cast(bbox_count[best_detect] % max_bbox_per_scale, tf.int32)
      bboxes_xywh = tf.tensor_scatter_nd_update(bboxes_xywh, indices=[[best_detect, bbox_ind]], updates=[bbox_xywh])
      bbox_count = tf.tensor_scatter_nd_add(bbox_count, indices=[[best_detect]], updates=[1])

  # retrieve actual sizes of each label feature an box
  target_labels = {}
  target_bboxes = {}
  for i in range(len(strides)):
    target_labels[i] = label[i, :train_output_sizes[i], :train_output_sizes[i]]
  for i in range(len(strides)):
    target_bboxes[i] = bboxes_xywh[i]

  return target_labels, target_bboxes


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou
