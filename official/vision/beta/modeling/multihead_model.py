"""
Build multihead multitask vision models.

Multihead tasks enables different inputs, losses, metrics of outputs from
different heads, but same backbone. This implementation leverages on 
SegmentationModel's structure that takes arbitrary backbone, decoder 
and head.
"""
from typing import List, Dict, Text

from absl import logging
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling.multitask import base_model
from official.modeling.multitask import configs as multi_cfg
from official.vision.beta.configs import multitask_config
from official.vision.beta.modeling.segmentation_model import SegmentationModel
from official.vision.beta.modeling.decoders import factory as decoder_factory
from official.vision.beta.modeling.heads import segmentation_heads, classification_heads
from official.vision.beta.modeling import factory_multitask

layers = tf.keras.layers


def build_model(task_config: multi_cfg.MultiTaskConfig):
  """Builds multitask model."""
  input_specs = layers.InputSpec(
      shape=[None] + task_config.model.input_size)

  l2_weight_decay = task_config.model.l2_weight_decay
  # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
  # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
  # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
  l2_regularizer = (tf.keras.regularizers.l2(
      l2_weight_decay / 2.0) if l2_weight_decay else None)

  model = factory_multitask.build_multihead_model(
      input_specs=input_specs,
      task_config=task_config,
      l2_regularizer=l2_regularizer)

  return model


def build_submodel(
    norm_activation_config: hyperparams.Config,
    backbone: tf.keras.Model,
    input_specs: tf.keras.layers.InputSpec,
    submodel_config: multitask_config.Submodel,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds submodel for a subtask. Leverages on SegmentationModel's structure that 
  takes any arbitrary backbone, decoder and head."""
  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=submodel_config,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  head_config = submodel_config.head

  if isinstance(head_config, multitask_config.ImageClassificationHead):
    head = classification_heads.ClassificationHead(
        num_classes=submodel_config.num_classes,
        level=head_config.level,
        num_convs=head_config.num_convs,
        num_filters=head_config.num_filters,
        add_head_batch_norm=head_config.add_head_batch_norm,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        dropout_rate=head_config.dropout_rate,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  elif isinstance(head_config, multitask_config.SegmentationHead):
    head = segmentation_heads.SegmentationHead(
        num_classes=submodel_config.num_classes,
        level=head_config.level,
        num_convs=head_config.num_convs,
        prediction_kernel_size=head_config.prediction_kernel_size,
        num_filters=head_config.num_filters,
        upsample_factor=head_config.upsample_factor,
        feature_fusion=head_config.feature_fusion,
        low_level=head_config.low_level,
        low_level_num_filters=head_config.low_level_num_filters,
        activation=norm_activation_config.activation,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)
  else:
    raise NotImplementedError('%s head is not implemented yet.' %(type(head_config)))

  return SegmentationModel(backbone, decoder, head)
  

class MultiHeadMultiTaskModel(base_model.MultiTaskBaseModel):

  def __init__(self,
               backbone: tf.keras.Model,
               heads: List[hyperparams.Config],
               norm_activation_config: hyperparams.Config,
               l2_regularizer: tf.keras.regularizers.Regularizer = None,
               input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
                 shape=[None, None, None, 3]),
               init_checkpoint: str = '',
               quantized: bool = False,
               *args, **kwargs):
    
    self.backbone = backbone
    self._input_specs = input_specs
    self._norm_activation_config = norm_activation_config
    self._l2_regularizer = l2_regularizer
    self._init_checkpoint = init_checkpoint
    self._quantized = quantized

    head_configs = {}
    for head_config in heads:
      head_configs[head_config.name] = head_config
    self._heads = head_configs
    
    super().__init__(*args, **kwargs)

  def _instantiate_sub_tasks(self) -> Dict[Text, tf.keras.Model]:
    """Builds models for each task for training"""
    sub_tasks = {}

    for name, submodel_config in self._heads.items():
      sub_tasks[name] = build_submodel(
        norm_activation_config=self._norm_activation_config,
        backbone=self.backbone,
        input_specs=self._input_specs,
        submodel_config=submodel_config,
        l2_regularizer=self._l2_regularizer)

    return sub_tasks
  
  def initialize(self):
    """Optional function that loads a pre-train checkpoint.
    Called after model subtasks are generated."""

    if self._init_checkpoint is None:
      return

    ckpt_dir_or_file = self._init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring same backbone
    ckpt = tf.train.Checkpoint(backbone=self.backbone)
    status = ckpt.restore(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    self._instantiate_sub_tasks()

    # Restoring heads if necessary
    for name, submodel in self._sub_tasks.items():
      if self._heads[name].init_checkpoint_modules == 'all':
        ckpt = tf.train.Checkpoint(decoder=submodel.decoder, 
                                   head=submodel.head)
        status = ckpt.restore(ckpt_dir_or_file)
        status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                  ckpt_dir_or_file)

  def __call__(self, 
               inputs: tf.Tensor, 
               training: bool = None) -> Dict[Text, tf.Tensor]:
    """Builds the multiheaded model for export. Not for training. Training 
    duplicates the shared backbone."""

    outputs = {}
    for name, subtask in self._sub_tasks.items():
      outputs[name] = subtask(inputs)

    return outputs
