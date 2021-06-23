# Lint as: python3
"""Multi vision configuration definition."""

from typing import List

import dataclasses

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.core import config_definitions as cfg
from official.modeling.multitask import configs as multi_cfg
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders
from official.vision.beta.configs import backbones
from official.vision.beta.configs.image_classification import \
  DataConfig as ClassificationDataConfig, \
  Losses as ClassificationLosses, \
  Evaluation as ClassificationEvaluation  
from official.vision.beta.configs.semantic_segmentation import \
  DataConfig as SegmentationDataConfig, \
  Losses as SegmentationLosses, \
  Evaluation as SegmentationEvaluation, \
  SegmentationHead


@dataclasses.dataclass
class ImageClassificationHead(hyperparams.Config):
  """Image classification head config"""
  level: int = 6
  num_convs: int = 2
  num_filters: int = 256
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False
  dropout_rate: float = 0.0


@dataclasses.dataclass
class Submodel(hyperparams.Config):
  name: str = 'foo'
  num_classes: int = 0
  min_level: int = 3 # only for FPN or NASFPN
  max_level: int = 6 # only for FPN or NASFPN
  head: hyperparams.Config = SegmentationHead()
  decoder: decoders.Decoder = decoders.Decoder(type='identity')
  init_checkpoint_modules: str = None


@dataclasses.dataclass
class MultiHeadModel(hyperparams.Config):
  """Multi head multi task model config, similar to other models but 
  with input, backbone, activation and weight decay shared."""
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  norm_activation: common.NormActivation = common.NormActivation()
  heads: List[Submodel] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  quantized: bool = False


@dataclasses.dataclass
class ImageClassificationModelSpecs(hyperparams.Config):
  """The model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ImageClassificationSubtask(cfg.TaskConfig):
  """The subtask config, similar to original task but with model and 
  its related parameters (init_checkpoint, init_checkpoint_moduless,
  model_output_keys) removed."""
  
  # passed into orbit.utils.make_distributed_dataset in base_trainer
  model: ImageClassificationModelSpecs = ImageClassificationModelSpecs()
  train_data: ClassificationDataConfig = ClassificationDataConfig(is_training=True)
  validation_data: ClassificationDataConfig = ClassificationDataConfig(is_training=False)
  losses: ClassificationLosses = ClassificationLosses() # used in task
  evaluation: ClassificationEvaluation = ClassificationEvaluation()


@dataclasses.dataclass
class SemanticSegmentationModelSpecs(hyperparams.Config):
  """The model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SemanticSegmentationSubtask(cfg.TaskConfig):
  """The subtask config, similar to original task but with model and 
  its related parameters (init_checkpoint, init_checkpoint_moduless,
  model_output_keys) removed."""
  model: SemanticSegmentationModelSpecs = SemanticSegmentationModelSpecs()
  train_data: SegmentationDataConfig = SegmentationDataConfig(is_training=True)
  validation_data: SegmentationDataConfig = SegmentationDataConfig(is_training=False)
  losses: SegmentationLosses = SegmentationLosses()
  evaluation: SegmentationEvaluation = SegmentationEvaluation()
  train_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)


@exp_factory.register_config_factory('multitask_vision')
def multitask_vision() -> multi_cfg.MultiTaskExperimentConfig:
  """
  Vision task with single backbone and multiple heads.
  Each head can be a segmenter, detector or classifier.
  TODO: support n heads instead of just one segmenter, one classifier.
  TODO: use same num_class and input_size in both task and model definition
  TODO: check where eval steps is used

  multi_cfg.MultiTaskConfig:
    - Retains each task_name, entire task, eval_steps and weights,
        - Entire_task used in respective multitask trainers for train_step
        - Weights used in task_sampler
  
  multi_cfg.MultiTaskTrainerConfig:
    - trainer_type and task_sampler used to configure task sampling in train_lib
    - Normal multi_cfg.TrainerConfig params used directly in train_lib
  """
  input_path_segmentation = ''
  input_path_classification = ''
  steps_per_epoch = 6915
  train_batch_size = 1
  eval_batch_size = 1
  validation_steps = 6915

  segmentation_routine = multi_cfg.TaskRoutine(
    task_name='segmentation',
    task_config=SemanticSegmentationSubtask(
      model=SemanticSegmentationModelSpecs(
          num_classes=19,
          input_size=[256, 256, 3]),
      losses=SegmentationLosses(
          ignore_label=250,
          top_k_percent_pixels=0.3),
      train_data=SegmentationDataConfig(
          output_size=[256, 256],
          input_path=input_path_segmentation,
          global_batch_size=train_batch_size,
          is_training=True,
          aug_scale_min=0.5,
          aug_scale_max=2.0,
          preserve_aspect_ratio=False,
          aug_policy='randaug',
          randaug_magnitude=5,
          randaug_available_ops=[
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 
            'Cutout', 'SolarizeAdd']),
      validation_data=SegmentationDataConfig(
          output_size=[256, 256],
          input_path=input_path_segmentation,
          global_batch_size=eval_batch_size,
          is_training=False,
          resize_eval_groundtruth=True,
          drop_remainder=False)),
    eval_steps=None, # check where eval steps is used
    task_weight=1.0
  )
  classification_routine = multi_cfg.TaskRoutine(
    task_name='classification',
    task_config=ImageClassificationSubtask(
      model=ImageClassificationModelSpecs(
          num_classes=4,
          input_size=[256, 256, 3]),
      losses=ClassificationLosses(
          label_smoothing=0.1),
      train_data=ClassificationDataConfig(
          input_path=input_path_classification,
          is_training=True,
          global_batch_size=train_batch_size,
          aug_policy='randaug',
          randaug_magnitude=5
      ),
      validation_data=ClassificationDataConfig(
          input_path=input_path_classification,
          is_training=False,
          global_batch_size=eval_batch_size,
          drop_remainder=False)
    ),
    eval_steps=None, # check where eval steps is used
    task_weight=1.0
  )
  
  model_config = MultiHeadModel(
    input_size=[256, 256, 3],
    backbone=backbones.Backbone(
        type='hardnet', hardnet=backbones.HardNet(model_id=70)),
    norm_activation=common.NormActivation(
        activation='relu',
        norm_momentum=0.9997,
        norm_epsilon=0.001,
        use_sync_bn=True),
    heads=[
      Submodel(
        name='classification',
        num_classes=4,
        head=ImageClassificationHead(
            level=0, # decoder is identity function
            num_convs=2,
            num_filters=256,
            add_head_batch_norm=False,
            dropout_rate=0.2)),
      Submodel(
        name='segmentation',
        num_classes=19,
        decoder=decoders.Decoder(
            type='hardnet', hardnet=decoders.HardNet(model_id=70)),
        head=SegmentationHead(
            level=0,
            num_convs=0,
            feature_fusion=None,
            low_level=0,
            low_level_num_filters=0))
    ],
    l2_weight_decay=1e-4
  )
  
  return multi_cfg.MultiTaskExperimentConfig(
    task=multi_cfg.MultiTaskConfig(
      model=model_config,
      init_checkpoint="",
      task_routines=(segmentation_routine, classification_routine)
    ), 
    trainer=multi_cfg.MultiTaskTrainerConfig(
      trainer_type="interleaving",
      task_sampler=multi_cfg.TaskSamplingConfig(type="proportional",
        proportional = multi_cfg.ProportionalSampleConfig(alpha=1.0)
      ), # uniform, proportional or annealing
      steps_per_loop=steps_per_epoch,
      summary_interval=steps_per_epoch,
      checkpoint_interval=steps_per_epoch,
      train_steps=45 * steps_per_epoch,
      validation_steps=validation_steps,
      validation_interval=steps_per_epoch,
      best_checkpoint_eval_metric= 'mean_iou',
      continuous_eval_timeout= 3600,
      max_to_keep= 5,
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
                  'decay_steps': 45 * steps_per_epoch,
                  'end_learning_rate': 0.0,
                  'power': 0.9
              }
          },
          'warmup': {
              'type': 'linear',
              'linear': {
                  'warmup_steps': 5 * steps_per_epoch,
                  'warmup_learning_rate': 0
              }
          }
      })
    )
  )
