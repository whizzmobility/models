"""YOLO task definition."""
from typing import Any, Optional, List, Tuple, Mapping, Union

from absl import logging
import numpy as np
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.vision.beta.configs import yolo as exp_cfg
from official.vision.beta.configs import multitask_config
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.dataloaders import yolo_input
from official.vision.beta.losses import yolo_losses
from official.vision.beta.modeling import factory
from official.vision.beta.ops import box_ops
from orbit.utils import SummaryManager
from official.vision.beta.evaluation import yolo_metrics


@task_factory.register_task_cls(multitask_config.YoloSubtask)
@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A task for YOLO."""

  def __init__(self, params, logging_dir: str = None, name: str = None):
    super().__init__(params, logging_dir, name)
    self.image_summary_manager = SummaryManager(self.logging_dir, tf.summary.image)

  def build_model(self):
    """Builds YOLO model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_yolo_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    return model

  def initialize(self, model: tf.keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    # check if path supplied is a file or directory of checkpoints
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if 'all' in self.task_config.init_checkpoint_modules:
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds yolo input."""

    if params.tfds_name:
      raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
    else:
      decoder = yolo_input.Decoder(is_bbox_in_pixels=params.is_bbox_in_pixels,
                                   is_xywh=params.is_xywh)

    model_params = self.task_config.model

    parser = yolo_input.Parser(
        output_size=params.output_size,
        input_size=model_params.input_size,
        anchor_per_scale=model_params.head.anchor_per_scale,
        num_classes=model_params.num_classes,
        max_bbox_per_scale=params.max_bbox_per_scale,
        strides=model_params.head.strides,
        anchors=model_params.head.anchors,
        aug_policy=params.aug_policy,
        randaug_magnitude=params.randaug_magnitude,
        randaug_available_ops=params.randaug_available_ops,
        aug_rand_hflip=params.aug_rand_hflip,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        preserve_aspect_ratio=params.preserve_aspect_ratio,
        aug_jitter_im=params.aug_jitter_im,
        aug_jitter_boxes=params.aug_jitter_boxes,
        dtype=params.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   model_outputs: Union[Mapping[str, tf.Tensor], tf.Tensor],
                   labels: Union[Mapping[str, tf.Tensor], tf.Tensor],
                   aux_losses: Optional[Any] = None):
    """YOLOv4 loss.

    Args:
      outputs: `dict` with 
        'raw_outputs', `dict[str, tf.Tensor]`, raw logits from final convolution and
        `prediction`, `dict[str, tf.Tensor]`, actual predictions for each scale
      labels: `dict` with
        `labels`, `dict[str, tf.Tensor]`, labels scaled according to feature size
        `bboxes`, `dict[str, tf.Tensor]`, bboxes scaled according to feature size
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    model_params = self._task_config.model
    loss_params = self._task_config.losses

    yolo_loss_fn = yolo_losses.YoloLoss(
        input_size=model_params.input_size[0],
        num_classes=model_params.num_classes,
        iou_loss_thres=loss_params.iou_loss_thres)

    total_giou_loss = total_conf_loss = total_prob_loss = 0

    for pred, conv, label, bboxes in zip(
      model_outputs['predictions'].values(), model_outputs['raw_outputs'].values(),
      labels['labels'].values(), labels['bboxes'].values()):
      
      giou_loss, conf_loss, prob_loss = yolo_loss_fn(
        pred=pred, conv=conv, label=label, bboxes=bboxes)
      total_giou_loss += giou_loss
      total_conf_loss += conf_loss
      total_prob_loss += prob_loss
    
    total_loss = total_giou_loss + total_conf_loss + total_prob_loss

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss, total_giou_loss, total_conf_loss, total_prob_loss

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    metrics = []
    metric_names = ['giou_loss', 'conf_loss', 'prob_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      metrics.append(yolo_metrics.AveragePrecisionAtIou(
        num_classes=self.task_config.model.num_classes, iou=0.25, name='AP25'
      ))
      metrics.append(yolo_metrics.AveragePrecisionAtIou(
        num_classes=self.task_config.model.num_classes, iou=0.5, name='AP50'
      ))

      # add in class specific metrics
      for class_num in range(self.task_config.model.num_classes):
        metrics.append(yolo_metrics.AveragePrecisionAtIou(
          num_classes=self.task_config.model.num_classes, iou=0.25, 
          name='precision_%s' %str(class_num),
          class_id=class_num
        ))
        metrics.append(yolo_metrics.AveragePrecisionAtIou(
          num_classes=self.task_config.model.num_classes, iou=0.5, 
          name='precision_%s' %str(class_num),
          class_id=class_num
        ))

    return metrics

  def train_step(self,
                 inputs: Tuple,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    input_partition_dims = self.task_config.train_input_partition_dims
    if input_partition_dims:
      strategy = tf.distribute.get_strategy()
      features = strategy.experimental_split_to_logical_devices(
          features, input_partition_dims)
    
    input_shape = self.task_config.model.input_size[:2]
    normalized_boxes = box_ops.normalize_boxes(labels['raw_bboxes'], input_shape)
    bbox_color = tf.constant([[1.0, 1.0, 0.0, 1.0]])
    self.image_summary_manager.write_summaries({
      'input_images': features,
      'bbox': tf.image.draw_bounding_boxes(features, normalized_boxes, bbox_color)
    })
    
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss, giou_loss, conf_loss, prob_loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)
    
    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    all_losses = {
      'giou_loss': giou_loss,
      'conf_loss': conf_loss,
      'prob_loss': prob_loss
    }
    if metrics:
      # process metrics uses labels and outputs, metrics.mean uses values only
      for m in metrics:
        m.update_state(all_losses[m.name])
        logs.update({m.name: m.result()})

    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf.keras.Model,
                      metrics: Optional[List[Any]] = None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    input_partition_dims = self.task_config.train_input_partition_dims
    if input_partition_dims:
      strategy = tf.distribute.get_strategy()
      features = strategy.experimental_split_to_logical_devices(
          features, input_partition_dims)
    
    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    loss, giou_loss, conf_loss, prob_loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)

    logs = {self.loss: loss}
    all_losses = {
      'giou_loss': giou_loss,
      'conf_loss': conf_loss,
      'prob_loss': prob_loss
    }
    if metrics:
      # process metrics uses labels and outputs, metrics.mean uses values only
      for metric in metrics:
        if 'loss' in metric.name:
          metric.update_state(all_losses[metric.name])
        else:
          metric.update_state(labels['labels'], outputs['predictions'])
        logs.update({metric.name: metric.result()})

    return logs

  def inference_step(self, inputs: tf.Tensor, model: tf.keras.Model):
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(self, state=None, step_outputs=None):
    pass

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    pass
