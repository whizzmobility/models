"""Metrics for YOLO."""

from typing import Optional, Mapping

import tensorflow as tf

from official.vision.beta.ops import yolo_ops


class AveragePrecisionAtIou(tf.keras.metrics.Precision):
  """Compute Average Precision at given box IOU for YOLO
  Prediction and ground-truth labels are first filtered by IoU, precision is then
  calculated, defined as: true_positives / (true_positives + false_positives)
  
  The predictions are accumulated in a confusion matrix, weighted by `sample_weight` 
  and the metric is then calculated from it. Weights default to 1.
  """

  def __init__(self, 
               num_classes: int, 
               iou: float,
               name: Optional[str] = None, 
               class_id: Optional[int] = None,
               *args, **kwargs):
    """Constructs YOLO evaluator class.

    Args:
      num_classes: The possible number of labels the prediction task can have.
      iou: `float`, iou threshold for calculating bboxes
      class_id: Class index to calculate precision over.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: `str`, name of the metric instance..
    """
    super(AveragePrecisionAtIou, self).__init__(
        name=name, class_id=class_id, *args, **kwargs)
    self.iou_thres = iou
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    
    Returns:
      Overall Precision, filtered at given IoU.
    """
    bbox_true, prob_true = yolo_ops.concat_tensor_dict(tensor_dict=y_true,
                                                       num_classes=self.num_classes)
    bbox_pred, prob_pred = yolo_ops.concat_tensor_dict(tensor_dict=y_pred,
                                                       num_classes=self.num_classes)

    iou = yolo_ops.bbox_iou(bbox_true, bbox_pred)
    iou_mask = iou > self.iou_thres

    prob_true = tf.boolean_mask(prob_true, iou_mask)
    prob_true = tf.argmax(prob_true, axis=-1)
    prob_true = tf.one_hot(prob_true, self.num_classes, on_value=True, off_value=False)

    prob_pred = tf.boolean_mask(prob_pred, iou_mask)
    prob_pred = tf.argmax(prob_pred, axis=-1)
    prob_pred = tf.one_hot(prob_pred, self.num_classes, on_value=True, off_value=False)

    return super(AveragePrecisionAtIou, self).update_state(
        y_true=prob_true, y_pred=prob_pred, sample_weight=None)
