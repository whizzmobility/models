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
"""Semantic segmentation input and model functions for serving/inference."""

from typing import Callable

import tensorflow as tf
import numpy as np

from official.vision.beta.modeling import factory
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops.colormaps import get_colormap
from official.vision.beta.serving import export_base, run_lib


class SegmentationModule(export_base.ExportModule):
  """Segmentation Module."""

  def __init__(self, 
               argmax_outputs: bool = True, 
               visualise_outputs: bool = True, 
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._argmax_outputs = argmax_outputs
    self._visualise_outputs = argmax_outputs and visualise_outputs

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def _build_inputs(self, image):
    """Builds segmentation model inputs for serving."""

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=run_lib.IMAGENET_MEAN_RGB,
                                           scale=run_lib.IMAGENET_STDDEV_RGB)

    image, _ = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=self._input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        preserve_aspect_ratio=False)
    return image

  def serve(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      `mask`: Tensor holding segmentation output logits, or class mask according to
        self._argmax_outputs
      `mask_visualised`: Tensor holding visualised class mask. Assumes output is 
        argmaxed.
    """
    # Removing nest.map_structure, as it adds a while node that is not static
    if images.shape[0] > 1:
      with tf.device('cpu:0'):
        images = tf.cast(images, dtype=tf.float32)

        images = tf.nest.map_structure(
            tf.identity,
            tf.map_fn(
                self._build_inputs, elems=images,
                fn_output_signature=tf.TensorSpec(
                    shape=self._input_image_size + [3], dtype=tf.float32),
                parallel_iterations=32
                )
            )
    else:
      images = tf.cast(images, dtype=tf.float32)
      images = tf.squeeze(images)
      images = self._build_inputs(images)
      images = tf.expand_dims(images, axis=0)

    mask = self.inference_step(images)
    mask = tf.image.resize(mask, self._input_image_size, method='bilinear')
    processed_outputs = {}

    if self._argmax_outputs:
      mask = tf.math.argmax(mask, -1)
    processed_outputs['mask'] = mask

    if self._visualise_outputs and len(mask.shape) == 3:
      colormap = get_colormap(cmap_type='cityscapes_int')
      mask = tf.gather(colormap, tf.cast(tf.squeeze(mask), tf.int32))
      processed_outputs['mask_visualised'] = mask

    return processed_outputs

  def run(self,
          image_path_glob: str,
          output_dir: str,
          preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
          inference_fn: Callable[[tf.Tensor], tf.Tensor],
          visualise: bool = True,
          stitch_original: bool = True,
          save_logits_bin: bool = False,
          *args, **kwargs):
    """Runs inference graph for the model, for given directory of images
    
    Args:
      image_path_glob: `str`, path pattern for images
      output_dir: `str`, path to output logs
      preprocess_fn: `Callable`, takes image tensor of shape (1, height, 
        width, channels), produces altered image tensor of same shape
      inference_fn: `Callable`, takes image tensor of shape (1, height, 
        width, channels), outputs Tensor of shape [batch_size, None, None, 3]
      visualise: `bool`, flag to use colormap
      stitch_original: `bool`, flag to stitch original image by the side
      save_logits_bin: `bool`, flag to save tensors and binary files
    """

    cmap = get_colormap(cmap_type='cityscapes').numpy()
    dataset = run_lib.inference_dataset(image_path_glob=image_path_glob,
                                        output_dir=output_dir,
                                        preprocess_fn=preprocess_fn)
    
    for image, img_filename, save_basename in dataset:

      logits = inference_fn(image)
      mask, visualised_mask = logits

      if save_logits_bin:
        run_lib.write_tensor_as_bin(tensor=image, 
                                    output_path=save_basename + '_input')
        run_lib.write_tensor_as_bin(tensor=mask, 
                                    output_path=save_basename + '_mask')
        run_lib.write_tensor_as_bin(tensor=visualised_mask, 
                                    output_path=save_basename + '_visualised_mask')

      mask = tf.squeeze(mask).numpy()
      if mask.ndim > 2:
          mask = np.argmax(mask, axis=-1).astype(np.uint8)

      if visualise:
        seg_map = cmap[mask]

      if stitch_original:
        image = tf.image.resize(image, seg_map.shape[:2])
        image = np.squeeze(image.numpy()).astype(np.uint8)
        seg_map = np.hstack((image, seg_map))
      
      encoded_seg_map = tf.image.encode_png(seg_map)
      tf.io.write_file(save_basename + '.png', encoded_seg_map)
      print("Visualised %s, saving result at %s" %(img_filename, save_basename + '.png'))
