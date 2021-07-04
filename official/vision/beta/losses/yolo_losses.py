"""Losses used for YOLO models."""

import tensorflow as tf

from official.vision.beta.ops import yolo_ops
from official.vision.beta.projects.yolo.ops import box_ops


class YoloLoss(tf.keras.losses.Loss):
  """Implements a YOLO loss for detection problems.
  Referenced from YOLOv4 implementation on:
    https://github.com/hunglc007/tensorflow-yolov4-tflite
  """

  def __init__(self,
               input_size: int,
               num_classes: int,
               iou_loss_thres: float,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='yolo_loss'):
    """Initializes `YOLOv4` loss. Consists of ciou, giou and diou.
      - Generalised IoU (considers shape, orientation in addition to overlap)
      - Center IoU (considers overlap, distance between center, aspect ratio)
      - Distance IoU (considers distance of center of bbox)

    Args:
      input_size: `int`, Dimension of input image
      num_classes: `int`, Number of foreground classes.
      iou_loss_thres: `float`, IoU threshold.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op. Defaults to 'yolo_loss'.
    """
    self.input_size = tf.cast(input_size, tf.float32)
    self.num_classes = num_classes
    self.iou_loss_thres = iou_loss_thres
    super(YoloLoss, self).__init__(reduction=reduction, name=name)

  def __call__(self, 
               pred: tf.Tensor, 
               conv: tf.Tensor, 
               label: tf.Tensor, 
               bboxes: tf.Tensor):
    """Invokes the `YoloLoss`.

    Args:
      pred: `tf.Tensor` of shape [batch, height, width, num_anchors, 5 + classes]
        denoting actual predictions for each scale
      conv: `tf.Tensor` of shape [batch, height, width, num_anchors, 5 + classes]
        denoting raw logits from final convolution
      label: `tf.Tensor` of shape [batch, height, width, num_anchors, 5 + classes]
        denoting groundtruth labels scaled according to feature size
      bboxes: `tf.Tensor` of shape [batch, height, width, num_anchors, 5 + classes]
        denoting ground truth bounding boxes scaled according to feature size

    Returns:
      giou_loss: Summed loss float `Tensor`,
      conf_loss: Summed loss float `Tensor`,
      prob_loss: Summed loss float `Tensor`
    """
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(box_ops.compute_giou(pred_xywh, label_xywh)[1], axis=-1)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (self.input_size * self.input_size)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = yolo_ops.bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thres, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        +
        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

  def get_config(self):
    config = {
        'strides': self.strides,
        'num_classes': self.num_classes,
        'iou_loss_thres': self.iou_loss_thres
    }
    base_config = super(YoloLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
