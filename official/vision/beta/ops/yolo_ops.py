"""Utility ops for yolo data.
Referenced from https://github.com/hunglc007/tensorflow-yolov4-tflite.
"""

from typing import List, Tuple, Mapping

import tensorflow as tf
import tensorflow_addons as tfa


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


def horizontal_flip_boxes(boxes, image_size):
  """Flips normalized boxes horizontally."""
  ymin, xmin, ymax, xmax = tf.split(
      value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(image_size, xmax)
  flipped_xmax = tf.subtract(image_size, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def random_horizontal_flip(image, box, seed=None):
  """Randomly flips input image and bounding boxes."""
  with tf.name_scope('random_horizontal_flip'):
    do_flip = tf.greater(tf.random.uniform([], seed=seed), 0.5)

    image = tf.cond(
        do_flip,
        lambda: tf.image.flip_left_right(image),
        lambda: image)
    
    image_size = tf.cast(tf.shape(image)[1], tf.float32)
    box = tf.cond(
        do_flip,
        lambda: horizontal_flip_boxes(box, image_size),
        lambda: box)

    return image, box


def random_translate(image, box, t, seed=None):
  """Randomly translate the image and boxes.

  Args:
      image: a `Tensor` representing the image.
      box: a `Tensor` represeting the boxes.
      t: an `int` representing the translation factor
      seed: an optional seed for tf.random operations
  Returns:
      image: a `Tensor` representing the augmented image.
      box: a `Tensor` representing the augmented boxes.
  """
  t_x = tf.random.uniform(minval=-t,
                          maxval=t,
                          shape=(),
                          dtype=tf.float32,
                          seed=seed)
  t_y = tf.random.uniform(minval=-t,
                          maxval=t,
                          shape=(),
                          dtype=tf.float32,
                          seed=seed)
  image_size = tf.cast(tf.shape(image)[1], tf.float32)
  with tf.name_scope('translate_boxes'):
    offset = tf.stack([t_y, t_x, t_y, t_x], axis=-1)
    box += offset * image_size
  with tf.name_scope('translate_image'):
    if (t_x != 0 and t_y != 0):
      image_jitter = tf.convert_to_tensor([t_x, t_y])
      image_jitter.set_shape([2])
      image = tfa.image.translate(image, image_jitter * image_size)
  return image, box


def preprocess_true_boxes(bboxes: tf.Tensor,
                          train_output_sizes: List[int],
                          anchor_per_scale: int,
                          num_classes: int,
                          max_bbox_per_scale: int,
                          strides: List[int],
                          anchors: tf.Tensor):
  """
  Args:
    bboxes: `tf.Tensor` of shape (None, 5), denoting (x, y, w, h, class), non-normalized
    train_output_sizes: `List[int]`, dimension of each scaled feature map
    anchor_per_scale: `int`, number of anchors per scale
    num_classes: `int`, number of classes.
    max_bbox_per_Scale: `int`, maximum number of bounding boxes per scale.
    strides: `List[int]` of output strides, ratio of input to output resolution.
      scaling of target feature depends on output sizes predicted by output strides
    anchors: `tf.Tensor` of shape (None, anchor_per_scale, 2) denothing positions
      of anchors

  !!! Assumes the images and boxes are preprocessed to fit image size.
  """

  max_output_size = tf.reduce_max(train_output_sizes)
  label = tf.zeros((len(strides), max_output_size, max_output_size, anchor_per_scale, 5+num_classes))

  bboxes_xywh = tf.zeros((len(strides), max_bbox_per_scale, 4))
  bbox_count = tf.zeros((3,))
  const = tf.constant([1.0], dtype=tf.float32)

  for bbox in bboxes:
    bbox_xywh = tf.cast(bbox[:4], tf.float32)
    bbox_class_ind = tf.cast(bbox[4], tf.int64)

    smooth_onehot = tf.one_hot(
      bbox_class_ind, num_classes, 
      on_value=(num_classes-1)/num_classes, 
      off_value=1.0/num_classes)

    bbox_xywh_scaled = tf.repeat(
      bbox_xywh[tf.newaxis, :], repeats=len(strides), axis=0)
    bbox_xywh_scaled /= [[i] for i in strides]

    bbox_label = tf.concat([bbox_xywh, const, smooth_onehot], axis=-1)

    iou = []
    exist_positive = False
    
    # register for each stride and corresponding anchor setting
    for i in range(3):
      # get anchor bbox in xywh format
      anchors_xywh = tf.add(tf.floor(bbox_xywh_scaled[i, 0:2]), 0.5)
      anchors_xywh = tf.repeat(anchors_xywh[tf.newaxis, :], 3, axis=0) 
      anchors_xywh = tf.concat([anchors_xywh, tf.cast(anchors[i], tf.float32)], axis=-1)

      # calculate iou for each anchor in this stride
      iou_scale = bbox_iou(bbox_xywh_scaled[i][tf.newaxis, :], anchors_xywh)
      iou.append(iou_scale)
      iou_mask = iou_scale > 0.3

      # update label at corresponding coordinate and boxes
      if tf.reduce_any(iou_mask):
        xind = tf.cast(tf.floor(bbox_xywh_scaled[i, 0]), tf.int32)
        yind = tf.cast(tf.floor(bbox_xywh_scaled[i, 1]), tf.int32)

        update = tf.gather([tf.zeros_like(bbox_label), bbox_label], tf.cast(iou_mask, tf.int64))
        label = tf.tensor_scatter_nd_update(label, indices=[[i, yind, xind]], updates=[update])

        bbox_ind = tf.cast(bbox_count[i] % max_bbox_per_scale, tf.int32)
        bboxes_xywh = tf.tensor_scatter_nd_update(bboxes_xywh, indices=[[i, bbox_ind]], updates=[bbox_xywh])
        bbox_count = tf.tensor_scatter_nd_add(bbox_count, indices=[[i]], updates=[1])

        exist_positive = True

    # registers for best anchor if bbox not registered
    if not exist_positive:
      best_anchor_ind = tf.argmax(tf.concat(iou, axis=-1), axis=-1) # anchor with highest iou
      best_detect = tf.cast(best_anchor_ind / anchor_per_scale, tf.int32) # corresponding stride level
      best_anchor = tf.cast(best_anchor_ind % anchor_per_scale, tf.int32) # anchor idx within stride
      
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

    union_area = bboxes1_area + bboxes2_area - inter_area + 1e-7

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def concat_tensor_dict(tensor_dict: Mapping[str, tf.Tensor],
                       num_classes: int):
  """Collate bbox and corresponding class tensors, from dictionary of tensors
  
  Args:
    tensor: `dict` with `tf.Tensor` values, of shape [batch, output_size, 
      output_size, anchors_per_scale, 5 + classes]
    num_classes: `int`, number of classes
  
  Returns:
    `bbox`: `tf.Tensor` of shape [batch, None, 4]
    `classes`: `tf.Tensor` of shape [batch, None, 1]
  """
  bbox_tensors = []
  prob_tensors = []

  for _, prediction in tensor_dict.items():
    pred_xywh, pred_conf, pred_prob = tf.split(prediction, (4, 1, num_classes), axis=-1)
    tensor_shape = pred_prob.shape
    num_instance = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (-1, num_instance, num_classes))
    pred_xywh = tf.reshape(pred_xywh, (-1, num_instance, 4))

    bbox_tensors.append(pred_xywh)
    prob_tensors.append(pred_prob)

  bbox_tensors = tf.concat(bbox_tensors, axis=1)
  prob_tensors = tf.concat(prob_tensors, axis=1)
  
  return bbox_tensors, prob_tensors


def filter_boxes(box_xywh: tf.Tensor, 
                 scores: tf.Tensor, 
                 score_threshold: float, 
                 input_shape: tf.Tensor):
    """Filter out boxes according to score threshold

    Args:
      box_xywh: `tf.Tensor`, of shape (batch size, None, 4), each entry being
        (centre x, centre y, width, height) of bbox
      scores: `tf.Tensor`, of shape (batch size, None, 6), denoting probabilities
        of being each class
      score_threshold: `float`, threshold to filter with
      input_shape: `tf.Tensor` denoting (height width) of image
    
    Returns:
      `boxes`: valid bounding boxes after filtered, shape (1, None, 4)
      `conf`: confidence score corresponding to bounding boxes, shape (1, None, classes)
    """
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    return boxes, pred_conf
