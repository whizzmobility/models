# Lint as: python3
"""YOLO input and model functions for serving/inference."""
from typing import Callable

import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import factory
from official.vision.beta.ops import box_ops, preprocess_ops, yolo_ops
from official.vision.beta.serving import export_base, run_lib
from official.vision.beta.projects.yolo.ops import box_ops as yolo_box_ops


class YoloModule(export_base.ExportModule):
  """YOLO Module."""

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_yolo_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def _build_inputs(self, image):
    """Builds YOLO model inputs for serving."""

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
      Tensor holding classification output logits.
    """
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

    outputs = self.inference_step(images) # tf.keras.Model's __call__ method

    num_classes = outputs['predictions']['0'].shape[-1] - 5
    bbox_tensors, prob_tensors = yolo_ops.concat_tensor_dict(
      tensor_dict=outputs['predictions'], 
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

    return {
      'boxes': boxes,
      'classes': classes,
      'scores': scores
    }

  def run_on_image_dir(self,
                       image_path_glob: str,
                       output_dir: str,
                       preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                       inference_fn: Callable[[tf.Tensor], tf.Tensor],
                       class_names_path: str,
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
      class_names_path: `str`, path to txt file containing classes. Text file
        should contain one class name per line.
      save_logits_bin: `bool`, flag to save tensors and binary files
    """

    dataset = run_lib.inference_dataset(image_path_glob=image_path_glob,
                                        output_dir=output_dir,
                                        preprocess_fn=preprocess_fn)
    class_names = run_lib.read_class_names(class_names_path=class_names_path)
    
    for image, img_filename, save_basename in dataset:

      logits = inference_fn(image)
      boxes, classes, scores = logits
      
      # multitensor output exporting is not deterministic
      if classes.dtype == 'float32':
        classes, scores = scores, classes

      if save_logits_bin:
        run_lib.write_tensor_as_bin(tensor=image, 
                                    output_path=save_basename + '_input')
        run_lib.write_tensor_as_bin(tensor=boxes, 
                                    output_path=save_basename + '_boxes')
        run_lib.write_tensor_as_bin(tensor=scores, 
                                    output_path=save_basename + '_scores')
        run_lib.write_tensor_as_bin(tensor=classes,
                                    output_path=save_basename + '_classes')

      image = tf.image.resize(image, self._input_image_size)
      image = tf.cast(image, tf.uint8)

      output_image = run_lib.draw_bbox(image=run_lib.tensor_to_numpy(image).squeeze(),
                                       bboxes=run_lib.tensor_to_numpy(boxes),
                                       scores=run_lib.tensor_to_numpy(scores),
                                       classes=run_lib.tensor_to_numpy(classes),
                                       num_bboxes=tf.constant([classes.shape[1]]).numpy(),
                                       class_names=class_names)
      
      output_image = tf.image.encode_png(output_image)
      tf.io.write_file(save_basename + '.png', output_image)
      print("Visualised %s, saving result at %s" %(img_filename, save_basename + '.png'))
