"""Metrics for YOLO."""

from typing import Optional, Mapping

import tensorflow as tf

from official.vision.beta.ops import yolo_ops


def applyIouAndConfThreshold(y_true: tf.Tensor,
                             y_pred: tf.Tensor,
                             num_classes: int,
                             conf_thres: float,
                             iou_thres: float):
  """
  Alter prediction and groundtruth tensors for update_state, according to confidence
  and iou threshold.

  COCO PrecisionAtIou logic:
    for each detection that has a confidence score > threshold:
      among the ground-truths, choose one that belongs to the same class 
      and has the highest IoU with the detection
      
      if no ground-truth can be chosen or IoU < threshold (e.g., 0.5):
        the detection is a false positive
      else:
        the detection is a true positive
  
  YoloV4 training is anchor boxes based, we choose the corresponding anchor box
  and coordinate position, for each detection, with sufficient confidence instead. Hence,
    1) set predictions to false when not enough confidence
    2) set groundtruth to false when enough confidence and not enough IoU

  TP: (1) enough confidence, (2) same class, (3) enough IoU
  FP: (1) enough confidence, (2) either wrong class or not enough IoU
  FN: (1) not enough confidence, (2) same class
  TN: (1) not enough confidence, (2) otherwise

  Args:
    y_true: The ground truth values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
    y_pred: The predicted values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
  """

  bbox_true, _, prob_true = yolo_ops.concat_tensor_dict(y_true, num_classes)
  bbox_pred, conf_pred, prob_pred = yolo_ops.concat_tensor_dict(y_pred, num_classes)

  conf_mask = conf_pred >= conf_thres
  conf_mask = tf.expand_dims(conf_mask, -1)
  iou = yolo_ops.bbox_iou(bbox_true, bbox_pred)
  lack_iou_mask = iou < iou_thres
  lack_iou_mask = tf.expand_dims(lack_iou_mask, -1)

  prob_pred = tf.argmax(prob_pred, axis=-1)
  prob_pred = tf.one_hot(prob_pred, num_classes, on_value=True, off_value=False)
  # retain predictions with sufficient confidence
  prob_pred = tf.where(conf_mask, prob_pred, tf.fill([num_classes], False))
  
  prob_true = tf.argmax(prob_true, axis=-1)
  prob_true = tf.one_hot(prob_true, num_classes, on_value=True, off_value=False)
  conf_and_no_iou_mask = tf.math.logical_and(conf_mask, lack_iou_mask)
  # remove ground truths with sufficient prediction confidence, not enough IoU
  prob_true = tf.where(conf_and_no_iou_mask, tf.fill([num_classes], False), prob_true)

  return prob_true, prob_pred


class AveragePrecisionAtIou(tf.keras.metrics.Precision):
  """Compute Average Precision at given box IOU for YOLO
  Precision is defined as: true_positives / (true_positives + false_positives)

  The predictions are accumulated in a confusion matrix, weighted by `sample_weight` 
  and the metric is then calculated from it. Weights default to 1.
  """

  def __init__(self, 
               num_classes: int, 
               iou: float,
               conf_thres: float = 0.3,
               name: Optional[str] = None, 
               class_id: Optional[int] = None,
               *args, **kwargs):
    """Constructs YOLO evaluator class.

    Args:
      num_classes: The possible number of labels the prediction task can have.
      iou: `float`, iou threshold for calculating bboxes
      conf_thres: `float`, confidence threshold to filter bboxes as per in COCO metrics
      class_id: Class index to calculate precision over.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: `str`, name of the metric instance..
    """
    super(AveragePrecisionAtIou, self).__init__(
        name=name, class_id=class_id, *args, **kwargs)
    self.conf_thres = conf_thres
    self.iou_thres = iou
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
      y_pred: The predicted values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    
    Returns:
      Overall Precision at IoU threshold
    """
    prob_true, prob_pred = applyIouAndConfThreshold(y_true=y_true,
                                                    y_pred=y_pred,
                                                    num_classes=self.num_classes,
                                                    conf_thres=self.conf_thres,
                                                    iou_thres=self.iou_thres)

    return super(AveragePrecisionAtIou, self).update_state(
        y_true=prob_true, y_pred=prob_pred, sample_weight=None)


class AverageRecallAtIou(tf.keras.metrics.Recall):
  """Compute Average Recall at given box IOU for YOLO
  Recall is defined as: true_positives / (true_positives + false_negative)

  The predictions are accumulated in a confusion matrix, weighted by `sample_weight` 
  and the metric is then calculated from it. Weights default to 1.
  """

  def __init__(self, 
               num_classes: int, 
               iou: float,
               conf_thres: float = 0.3,
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
    super(AverageRecallAtIou, self).__init__(
        name=name, class_id=class_id, *args, **kwargs)
    self.conf_thres = conf_thres
    self.iou_thres = iou
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
      y_pred: The predicted values, shape (batch, yind, xind, anchorind, xywh/conf/classes)
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    
    Returns:
      Overall Recall, filtered at given IoU.
    """
    prob_true, prob_pred = applyIouAndConfThreshold(y_true=y_true,
                                                    y_pred=y_pred,
                                                    num_classes=self.num_classes,
                                                    conf_thres=self.conf_thres,
                                                    iou_thres=self.iou_thres)

    return super(AverageRecallAtIou, self).update_state(
        y_true=prob_true, y_pred=prob_pred, sample_weight=None)
