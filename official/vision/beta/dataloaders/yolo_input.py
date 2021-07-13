"""Data parser and processing for yolo datasets."""
from typing import Optional, List

import tensorflow as tf
import numpy as np

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import augment, yolo_ops, box_ops, preprocess_ops
from official.vision.beta.projects.yolo.ops import box_ops as yolo_box_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Decoder(decoder.Decoder):
  """A tf.Example decoder for yolo task."""

  def __init__(self, 
               is_bbox_in_pixels=False, 
               is_xywh=False):
    self._keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'bbox/class':tf.io.VarLenFeature(tf.float32),
        'bbox/x':tf.io.VarLenFeature(tf.float32),
        'bbox/y':tf.io.VarLenFeature(tf.float32),
        'bbox/w':tf.io.VarLenFeature(tf.float32),
        'bbox/h':tf.io.VarLenFeature(tf.float32)
    }

    self.is_bbox_in_pixels = is_bbox_in_pixels
    self.is_xywh = is_xywh
  
  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    image = tf.reshape(image, (height, width, 3))
    return image

  def _decode_boxes(self, parsed_tensors):
    """Concat box coordinates in the format of [x, y, width, height]."""
    x = parsed_tensors['bbox/x']
    y = parsed_tensors['bbox/y']
    w = parsed_tensors['bbox/w']
    h = parsed_tensors['bbox/h']

    if not self.is_bbox_in_pixels:
      x = x * tf.cast(parsed_tensors['image/width'], tf.float32)
      y = y * tf.cast(parsed_tensors['image/height'], tf.float32)
      w = w * tf.cast(parsed_tensors['image/width'], tf.float32)
      h = h * tf.cast(parsed_tensors['image/height'], tf.float32)
    
    bbox = tf.stack([x, y, w, h], axis=-1)
    if self.is_xywh:
      bbox = yolo_box_ops.xcycwh_to_yxyx(bbox)

    return bbox

  def decode(self, serialized_example):
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._keys_to_features)
    
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)
    
    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    decoded_tensors = {
        'image': image,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'boxes': boxes,
        'classes': parsed_tensors['bbox/class']
    }

    return decoded_tensors


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors.
  """

  def __init__(self,
               output_size,
               input_size: List[int],
               anchor_per_scale: int,
               num_classes: int,
               max_bbox_per_scale: int,
               strides: List,
               anchors: List,
               aug_policy: Optional[str] = None,
               randaug_magnitude: Optional[int] = 10,
               randaug_available_ops: Optional[List[str]] = None,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               preserve_aspect_ratio=True,
               aug_jitter_im=0.1,
               aug_jitter_boxes=0.005,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    !!! Augmentation ops assumes that boxes are yxyx format, non-normalized
      (top left, bottom right coords) in pixels.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      input_size: `List[int]`, shape of image input
      anchor_per_scale: `int`, number of anchors per scale
      num_classes: `int`, number of classes.
      max_bbox_per_Scale: `int`, maximum number of bounding boxes per scale.
      strides: `List[int]` of output strides, ratio of input to output resolution.
      anchors: `tf.Tensor` of shape (None, anchor_per_scale, 2) denothing positions
        of anchors
      aug_policy: `str`, augmentation policies. None or 'randaug'. TODO support 'autoaug'
      randaug_magnitude: `int`, magnitude of the randaugment policy.
      randaug_available_ops: `List[str]`, specify augmentations for randaug
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      preserve_aspect_ratio: `bool`, whether to preserve aspect ratio during resize
      aug_jitter_im: `float`, pixel value of maximum jitter applied to the image
      aug_jitter_boxes: `float`, pixel value of maximum jitter applied to bbox
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._output_size = output_size
    self._input_size = input_size

    # yolo true boxes processing
    self.train_output_sizes = input_size[0] // np.array(strides)
    self.anchor_per_scale = anchor_per_scale
    self.num_classes = num_classes
    self.max_bbox_per_scale = max_bbox_per_scale
    self.strides = strides
    self.anchors = tf.constant(anchors, dtype=tf.float32)
    self.anchors = tf.reshape(self.anchors, [
      int(len(anchors)/self.anchor_per_scale/2), self.anchor_per_scale, 2])

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._preserve_aspect_ratio = preserve_aspect_ratio
    self._aug_jitter_im = aug_jitter_im
    self._aug_jitter_boxes = aug_jitter_boxes

    if aug_policy:
      # ops that changes the shape of the mask (any form of translation / rotation)
      if aug_policy == 'randaug':
        self._augmenter = augment.RandAugment(
            num_layers=2, magnitude=randaug_magnitude, available_ops=randaug_available_ops)
      else:
        raise ValueError(
            'Augmentation policy {} not supported.'.format(aug_policy))
    else:
      self._augmenter = None

    # dtype.
    self._dtype = dtype

  def _parse_train_data(self, data):
    """Parses data for training and evaluation.
    !!! All augmentations and transformations are on bboxes with format
      (ymin, xmin, ymax, xmax). Required to do the appropriate transformations.
    !!! Images are supposed to be in RGB format
    """
    image, boxes = data['image'], data['boxes']

    # Execute RandAugment first as some ops require uint8 colors
    if self._augmenter is not None:
      image = self._augmenter.distort(image)

    if self._aug_rand_hflip:  
      image, boxes = yolo_ops.random_horizontal_flip(image, boxes)
    
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._input_size[:2],
        self._input_size[:2],
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max,
        preserve_aspect_ratio=self._preserve_aspect_ratio)
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_info[2, :],
                                                 image_info[1, :], image_info[3, :])

    if self._aug_jitter_im != 0.0:
      image, boxes = yolo_ops.random_translate(image, boxes, self._aug_jitter_im)

    if self._aug_jitter_boxes != 0.0:
      boxes = box_ops.jitter_boxes(boxes, self._aug_jitter_boxes)

    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)
    image = tf.cast(image, dtype=self._dtype)

    boxes = tf.clip_by_value(boxes, 0, self._input_size[0]-1)
    bbox_labels = yolo_box_ops.yxyx_to_xcycwh(boxes)
    bbox_labels = tf.concat([bbox_labels, data['classes'][:, tf.newaxis]], axis=-1)

    labels, bbox_labels = yolo_ops.preprocess_true_boxes(
      bboxes=bbox_labels,
      train_output_sizes=self.train_output_sizes,
      anchor_per_scale=self.anchor_per_scale,
      num_classes=self.num_classes,
      max_bbox_per_scale=self.max_bbox_per_scale,
      strides=self.strides,
      anchors=self.anchors)
    
    # pad / limit to 10 boxes for constant size
    raw_bboxes = boxes
    num_bboxes = tf.shape(raw_bboxes)[0]
    if num_bboxes > 10:
      raw_bboxes = raw_bboxes[:, :10]
    else:
      paddings = tf.stack([0, 10-num_bboxes], axis=-1)
      paddings = tf.stack([paddings, [0,0]], axis=0)
      raw_bboxes = tf.pad(raw_bboxes, paddings)

    targets = {
      'labels': labels,
      'bboxes': bbox_labels,
      'raw_bboxes': raw_bboxes
    }

    return image, targets

  def _parse_eval_data(self, data):
    """Parses data for evaluation.
    !!! All augmentations and transformations are on bboxes with format
      (ymin, xmin, ymax, xmax). Required to do the appropriate transformations.
    !!! Images are supposed to be in RGB format
    """
    image, boxes = data['image'], data['boxes']

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._input_size[:2],
        self._input_size[:2],
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max,
        preserve_aspect_ratio=self._preserve_aspect_ratio)
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_info[2, :],
                                                 image_info[1, :], image_info[3, :])

    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)
    image = tf.cast(image, dtype=self._dtype)

    boxes = tf.clip_by_value(boxes, 0, self._input_size[0]-1)
    bbox_labels = yolo_box_ops.yxyx_to_xcycwh(boxes)
    bbox_labels = tf.concat([bbox_labels, data['classes'][:, tf.newaxis]], axis=-1)

    labels, bbox_labels = yolo_ops.preprocess_true_boxes(
      bboxes=bbox_labels,
      train_output_sizes=self.train_output_sizes,
      anchor_per_scale=self.anchor_per_scale,
      num_classes=self.num_classes,
      max_bbox_per_scale=self.max_bbox_per_scale,
      strides=self.strides,
      anchors=self.anchors)
    
    targets = {
      'labels': labels,
      'bboxes': bbox_labels
    }

    return image, targets
