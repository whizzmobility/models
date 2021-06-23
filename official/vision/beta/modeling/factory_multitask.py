"""Factory methods to build models."""

# Import libraries
import tensorflow as tf

from official.modeling.multitask import configs as multi_cfg
from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import multihead_model


def build_multihead_model(
    input_specs: tf.keras.layers.InputSpec,
    task_config: multi_cfg.MultiTaskConfig,
    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  
  model_config = task_config.model
  norm_activation_config = model_config.norm_activation
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  model = multihead_model.MultiHeadMultiTaskModel(
    backbone=backbone,
    heads=model_config.heads,
    norm_activation_config=norm_activation_config,
    l2_regularizer=l2_regularizer,
    input_specs=input_specs,
    init_checkpoint=task_config.init_checkpoint,
    quantized=model_config.quantized
  )

  return model
