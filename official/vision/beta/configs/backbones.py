# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Backbones configurations."""
from typing import Optional, List

# Import libraries
import dataclasses

from official.modeling import hyperparams


@dataclasses.dataclass
class ResNet(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50
  depth_multiplier: float = 1.0
  stem_type: str = 'v0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False


@dataclasses.dataclass
class ResNest(hyperparams.Config):
  """ResNest config."""
  model_id: int = 50
  stem_type: str = 'v0'


@dataclasses.dataclass
class DilatedResNet(hyperparams.Config):
  """DilatedResNet config."""
  model_id: int = 50
  output_stride: int = 16
  multigrid: Optional[List[int]] = None
  stem_type: str = 'v0'
  last_stage_repeats: int = 1
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class EfficientNet(hyperparams.Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class DilatedEfficientNet(hyperparams.Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  output_stride: int = 16
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class HardNet(hyperparams.Config):
  """HardNet config."""
  model_id: int = 70


@dataclasses.dataclass
class MobileNet(hyperparams.Config):
  """Mobilenet config."""
  model_id: str = 'MobileNetV2'
  filter_size_scale: float = 1.0
  stochastic_depth_drop_rate: float = 0.0
  output_stride: Optional[int] = None


@dataclasses.dataclass
class SpineNet(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  min_level: int = 3
  max_level: int = 7


@dataclasses.dataclass
class SpineNetMobile(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.2
  expand_ratio: int = 6
  min_level: int = 3
  max_level: int = 7


@dataclasses.dataclass
class RevNet(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: int = 56


@dataclasses.dataclass
class DarkNet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = "darknet53"


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    resnet: resnet backbone config.
    resnest: resnest backbone config.
    dilated_resnet: dilated resnet backbone for semantic segmentation config.
    revnet: revnet backbone config.
    efficientnet: efficientnet backbone config.
    dilated_efficientnet: dilated efficientnet backbone for semantic segmentation config.
    hardnet: hardnet backbone for semantic segmentation config
    spinenet: spinenet backbone config.
    spinenet_mobile: mobile spinenet backbone config.
    mobilenet: mobilenet backbone config.
  """
  type: Optional[str] = None
  freeze: bool = False

  resnet: ResNet = ResNet()
  resnest: ResNest = ResNest()
  dilated_resnet: DilatedResNet = DilatedResNet()
  revnet: RevNet = RevNet()
  efficientnet: EfficientNet = EfficientNet()
  dilated_efficientnet: DilatedEfficientNet = DilatedEfficientNet()
  hardnet: HardNet = HardNet()
  spinenet: SpineNet = SpineNet()
  spinenet_mobile: SpineNetMobile = SpineNetMobile()
  mobilenet: MobileNet = MobileNet()
  darknet: DarkNet = DarkNet()
