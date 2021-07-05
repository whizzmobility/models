"""Data parser and processing for yolo datasets."""
from typing import Optional, List

import tensorflow as tf
import numpy as np

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import augment, preprocess_ops, yolo_ops
from official.vision.beta.projects.yolo.ops import preprocess_ops as yolo_preprocess_ops


class Decoder(decoder.Decoder):
  """A tf.Example decoder for yolo task."""

  def __init__(self, is_bbox_in_pixels=False):
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
  
  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
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

    return tf.stack([x, y, w, h], axis=-1)

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
               is_bbox_in_pixels: bool,
               is_xywh: bool,
               aug_policy: Optional[str] = None,
               randaug_magnitude: Optional[int] = 10,
               randaug_available_ops: Optional[List[str]] = None,
               aug_rand_hflip=False,
               aug_jitter_im=0.1,
               aug_rand_saturation=True,
               aug_rand_brightness=True,
               aug_rand_zoom=True,
               aug_rand_hue=True,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

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
      is_bbox_in_pixels: `bool`, true if bounding box values are in pixels
      is_xywh: `bool`, true if bounding box values are in (x, y, width, height) format
      aug_policy: `str`, augmentation policies. None or 'randaug'. TODO support 'autoaug'
      randaug_magnitude: `int`, magnitude of the randaugment policy.
      randaug_available_ops: `List[str]`, specify augmentations for randaug
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_zoom: `bool`, if True, augment training with random zoom.
      aug_rand_hue: `bool`, if True, augment training with random hue.
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
    self.anchors = tf.reshape(self.anchors, [int(len(anchors)/6), 3, 2])
    self.is_bbox_in_pixels = is_bbox_in_pixels
    self.is_xywh = is_xywh

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_hue = aug_rand_hue
    self._aug_jitter_im = aug_jitter_im

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
    """Parses data for training and evaluation."""
    image, boxes = data['image'], data['boxes']
    image /= 255

    image, boxes = yolo_ops.resize_image_and_bboxes(
      image=image, 
      bboxes=boxes, 
      target_size=self._input_size[:2], 
      preserve_aspect_ratio=False,
      image_height=data['height'],
      image_width=data['width'],
      image_normalized=True)

    if self._aug_rand_hflip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)
    
    #TODO(ruien): implement random zoom

    if self._aug_jitter_im != 0.0:
      image, boxes = yolo_preprocess_ops.random_translate(
          image, boxes, self._aug_jitter_im)
    
    if self._aug_rand_brightness:
      image = tf.image.random_brightness(
          image=image, max_delta=.1)  # Brightness

    if self._aug_rand_saturation:
      image = tf.image.random_saturation(
          image=image, lower=0.75, upper=1.25)  # Saturation

    if self._aug_rand_hue:
      image = tf.image.random_hue(image=image, max_delta=.3)  # Hue

    image = tf.clip_by_value(image, 0.0, 1.0)
    boxes = tf.concat([boxes, data['classes'][:, tf.newaxis]], axis=-1)

    result = yolo_ops.preprocess_true_boxes(
      bboxes=boxes,
      train_output_sizes=self.train_output_sizes,
      anchor_per_scale=self.anchor_per_scale,
      num_classes=self.num_classes,
      max_bbox_per_scale=self.max_bbox_per_scale,
      strides=self.strides,
      anchors=self.anchors,
      is_bbox_in_pixels=self.is_bbox_in_pixels,
      is_xywh=self.is_xywh)

    return image, *result

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, bboxes = self._prepare_image_and_bbox(data)

    return image, bboxes
