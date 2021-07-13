# Lint as: python3
"""YOLO input and model functions for serving/inference."""
from typing import Callable

import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import factory
from official.vision.beta.ops import preprocess_ops, yolo_ops
from official.vision.beta.serving import export_base, run_lib


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


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

    bbox_tensors = tf.concat(bbox_tensors, axis=1)
    prob_tensors = tf.concat(prob_tensors, axis=1)

    return {
      'boxes': bbox_tensors,
      'pred_conf': prob_tensors
    }

  def run(self,
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
    
    for image, img_filename, save_basename in dataset:

      logits = inference_fn(image)
      bbox_tensors, prob_tensors = logits

      if save_logits_bin:
        run_lib.write_tensor_as_bin(tensor=image, 
                                    output_path=save_basename + '_input')
        run_lib.write_tensor_as_bin(tensor=bbox_tensors, 
                                    output_path=save_basename + '_boxes')
        run_lib.write_tensor_as_bin(tensor=prob_tensors, 
                                    output_path=save_basename + '_pred_conf')

      boxes, pred_conf = yolo_ops.filter_boxes(box_xywh=bbox_tensors,
                                               scores=prob_tensors,
                                               score_threshold=0.4,
                                               input_shape=self._input_image_size)

      boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (self._batch_size, -1, 1, 4)),
        scores=tf.reshape(pred_conf, (self._batch_size, -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.25
      )

      image_dim = min(image.shape[1], image.shape[2])
      image = tf.image.resize(image, [image_dim, image_dim])
      image = tf.cast(image, tf.uint8)
      image = image.numpy().squeeze()
      class_names = yolo_ops.read_class_names(class_names_path=class_names_path)

      output_image = yolo_ops.draw_bbox(image=image,
                                        bboxes=boxes.numpy(),
                                        scores=scores.numpy(),
                                        classes=classes.numpy(),
                                        num_bboxes=valid_detections.numpy(),
                                        class_names=class_names)
      
      output_image = tf.image.encode_png(output_image)
      tf.io.write_file(save_basename + '.png', output_image)
      print("Visualised %s, saving result at %s" %(img_filename, save_basename + '.png'))
