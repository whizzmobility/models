# Lint as: python3
"""Semantic segmentation input and model functions for serving/inference."""

from typing import Mapping, Callable

import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import factory_multitask
from official.vision.beta.ops import preprocess_ops, yolo_ops, box_ops
from official.vision.beta.ops.colormaps import get_colormap
from official.vision.beta.serving import export_base, run_lib
from official.vision.beta.projects.yolo.ops import box_ops as yolo_box_ops

# RGB mean and stddev from ImageNet
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class MultitaskModule(export_base.ExportModule):
  """Multitask Module."""

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

    return factory_multitask.build_multihead_model(
        input_specs=input_specs,
        task_config=self.params.task,
        l2_regularizer=None)

  def _build_inputs(self, image: tf.Tensor) -> tf.Tensor:
    """Builds classification model inputs for serving."""

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    image, _ = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=self._input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        preserve_aspect_ratio=False)
    return image

  def serve(self, images: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
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

    outputs = self.inference_step(images)
    processed_outputs = {}
    
    for name, output in outputs.items():
      
      if 'classification' in name:
        if self._argmax_outputs:
          output = tf.math.argmax(output, -1)
        else:
          output = tf.nn.softmax(output)
        processed_outputs[name] = output

      elif 'segmentation' in name:
        output = tf.image.resize(
          output, self._input_image_size, method='bilinear')      
        
        if self._argmax_outputs:
          output = tf.math.argmax(output, -1)
        processed_outputs[name] = output
        
        if self._visualise_outputs and len(output.shape) == 3:
          colormap = get_colormap(cmap_type='cityscapes_int')
          processed_outputs[name + '_visualised'] = tf.gather(
            colormap, tf.cast(tf.squeeze(output), tf.int32))
      
      elif 'yolo' in name:
        num_classes = output['predictions']['0'].shape[-1] - 5
        bbox_tensors, prob_tensors = yolo_ops.concat_tensor_dict(
          tensor_dict=output['predictions'], 
          num_classes=num_classes
        )

        boxes = tf.concat(bbox_tensors, axis=1)
        boxes = tf.squeeze(yolo_box_ops.xcycwh_to_yxyx(boxes))
        scores = tf.concat(prob_tensors, axis=1)
        scores = tf.squeeze(tf.math.reduce_max(scores, axis=-1))
        classes = tf.argmax(prob_tensors, axis=-1)
        
        indices = tf.image.non_max_suppression(boxes=boxes,
                                               scores=scores,
                                               max_output_size=20,
                                               iou_threshold=0.5,
                                               score_threshold=0.25)
        
        boxes = tf.expand_dims(tf.gather(boxes, indices), axis=0)
        boxes = box_ops.normalize_boxes(boxes, self._input_image_size)
        scores = tf.expand_dims(tf.gather(scores, indices), axis=0)
        classes = tf.gather(classes, indices, axis=1)

        processed_outputs[name + 'boxes'] = boxes
        processed_outputs[name + 'classes'] = classes
        processed_outputs[name + 'scores'] = scores

      else:
        raise NotImplementedError('Task type %s is not implemented.' + \
          'Try renaming the task routine.' %name)

    return processed_outputs
