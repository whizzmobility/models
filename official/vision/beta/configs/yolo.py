"""YOLO configuration definition."""

from typing import List, Optional, Union
import dataclasses

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  output_size: List[int] = dataclasses.field(default_factory=list)
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  aug_rand_hflip: bool = True
  aug_jitter_im: float = 0.1
  aug_rand_saturation: bool = True
  aug_rand_brightness: bool = True
  aug_rand_zoom: bool = True
  aug_rand_hue: bool = True
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'
  randaug_magnitude: Optional[int] = 10
  randaug_available_ops: Optional[List[str]] = None
  drop_remainder: bool = True
  file_type: str = 'tfrecord'

  max_bbox_per_scale: int = 150
  is_bbox_in_pixels: bool = True
  is_xywh: bool = False


@dataclasses.dataclass
class YoloHead(hyperparams.Config):
  anchor_per_scale: int = 3
  strides: List[int] = dataclasses.field(default_factory=list)
  anchors: List[int] = dataclasses.field(default_factory=list)
  xy_scale: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class YoloModel(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3 # only for FPN or NASFPN
  max_level: int = 6 # only for FPN or NASFPN
  head: hyperparams.Config = YoloHead()
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(type='identity')
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class YoloLosses(hyperparams.Config):
  label_smoothing: float = 0.0
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  iou_loss_thres: float = 0.5


@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
  """The model config."""
  model: YoloModel = YoloModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: YoloLosses = YoloLosses()
  train_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder


@exp_factory.register_config_factory('yolo')
def detector_yolo() -> cfg.ExperimentConfig:
  """YOLO on custom datasets"""
  
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
        run_eagerly=True
      ),
      task=YoloTask(
          model=YoloModel(
              num_classes=6,
              input_size=[256, 256, 3],
              backbone=backbones.Backbone(
                type='hardnet', hardnet=backbones.HardNet(model_id=70)),
              decoder=decoders.Decoder(
                  type='pan', pan=decoders.PAN(levels=3)),
              head=YoloHead(
                anchor_per_scale=3,
                strides=[16, 32, 64],
                anchors=[12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401],
                xy_scale=[1.2, 1.1, 1.05]
              ),
              norm_activation=common.NormActivation(
                activation='relu',
                norm_momentum=0.9997,
                norm_epsilon=0.001,
                use_sync_bn=True)),
          losses=YoloLosses(l2_weight_decay=1e-4,
                            iou_loss_thres=0.5),
          train_data=DataConfig(
              input_path='D:/data/whizz_tf/detect_env*',
              output_size=[256, 256],
              is_training=True,
              global_batch_size=1),
          validation_data=DataConfig(
              input_path='D:/data/whizz_tf/detect_env*',
              output_size=[256, 256],
              is_training=False,
              global_batch_size=1,
              drop_remainder=False),
          # init_checkpoint=None
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=2,
          summary_interval=2,
          checkpoint_interval=2,
          train_steps=20,
          validation_steps=20,
          validation_interval=2,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 20,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
