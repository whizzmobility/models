"""Metrics for classification."""

from absl import logging
import tensorflow as tf


class Precision(tf.keras.metrics.Precision):
  """Computes Precision metric using labels.
  Defined as: true_positives / (true_positives + false_positives)
  
  The predictions are accumulated in a confusion matrix, weighted by `sample_weight` 
  and the metric is then calculated from it. Weights default to 1.
  """

  def __init__(self, num_classes, class_id=None, name=None, *args, **kwargs):
    """Initializes `Precision`

    Args:
      num_classes: The possible number of labels the prediction task can have.
      class_id: Class index to calculate precision over.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Precision, self).__init__(
      name=name, class_id=class_id, *args, *kwargs)
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    
    Returns:
      Precision per class.
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, self.num_classes, on_value=True, off_value=False)
    
    return super(Precision, self).update_state(
        y_true=y_true, y_pred=y_pred, sample_weight=None)


class Recall(tf.keras.metrics.Recall):
  """Computes Recall metric using labels.
  Defined as: true_positives / (true_positives + false_negatives)
  
  The predictions are accumulated in a confusion matrix, weighted by `sample_weight` 
  and the metric is then calculated from it. Weights default to 1.
  """

  def __init__(self, num_classes, class_id=None, name=None, *args, **kwargs):
    """Initializes `Recall`

    Args:
      num_classes: The possible number of labels the prediction task can have.
      class_id: Class index to calculate recall over.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Recall, self).__init__(
      name=name, class_id=class_id, *args, *kwargs)
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    
    Returns:
      Recall per class.
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, self.num_classes, on_value=True, off_value=False)
    
    return super(Recall, self).update_state(
        y_true=y_true, y_pred=y_pred, sample_weight=None)
