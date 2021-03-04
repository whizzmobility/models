# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions of EfficientNetKerasApp Networks."""

import math
# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers

MODELS = {
  'b0': tf.keras.applications.EfficientNetB0,
  'b1': tf.keras.applications.EfficientNetB1,
  'b2': tf.keras.applications.EfficientNetB2,
  'b3': tf.keras.applications.EfficientNetB3,
  'b4': tf.keras.applications.EfficientNetB4,
  'b5': tf.keras.applications.EfficientNetB5,
  'b6': tf.keras.applications.EfficientNetB6,
  'b7': tf.keras.applications.EfficientNetB7
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class EfficientNetKerasApp(tf.keras.Model):
  """Class to build EfficientNetKerasApp family model."""

  def __init__(self,
               model_id,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               **kwargs):
    """EfficientNetKerasApp initialization using keras applications.

    Args:
      model_id: `str` model id of EfficientNetKerasApp.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
    """
    self._model_id = model_id
    self._input_specs = input_specs

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build EfficientNetKerasApp.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    model = MODELS[model_id](include_top=False, weights='imagenet', input_tensor=inputs)
    model.trainable = False
    endpoints = {'0': model(inputs)}

    # Build output specs for downstream tasks.
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints.keys()}

    for i, v in endpoints.items():
      print('endpoint', i, v, v.shape)
    for i, v in self.output_specs.items():
      print('output_spec item', i, v)

    super(EfficientNetKerasApp, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('efficientnetkerasapp')
def build_efficientnet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds ResNet 3d backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'efficientnetkerasapp', (f'Inconsistent backbone type '
                                           f'{backbone_type}')

  return EfficientNetKerasApp(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs)
