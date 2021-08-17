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


class MultitaskModule(export_base.ExportModule):
  """Multitask Module."""

  def __init__(self, 
               argmax_outputs: bool = True, 
               visualise_outputs: bool = True, 
               class_present_outputs: bool = True,
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._argmax_outputs = argmax_outputs
    self._visualise_outputs = argmax_outputs and visualise_outputs
    self._class_present_outputs = argmax_outputs and class_present_outputs

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
        num_classes = output.shape[-1]

        if self._class_present_outputs:
          flattened_output = tf.math.argmax(tf.reshape(output, [-1, num_classes]), -1)
          one_hotted = tf.one_hot(flattened_output, 19, axis=0)
          class_counts = tf.reduce_sum(one_hotted, axis=-1)
          processed_outputs[name + '_class_count'] = class_counts

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
        classes = tf.squeeze(tf.math.argmax(prob_tensors, axis=-1))
        
        indices = tf.image.non_max_suppression(boxes=boxes,
                                               scores=scores,
                                               max_output_size=20,
                                               iou_threshold=0.25,
                                               score_threshold=0.25)
        
        boxes = tf.gather(boxes, indices)
        scores = tf.gather(scores, indices)
        classes = tf.gather(classes, indices)

        processed_outputs[name + 'boxes'] = boxes
        processed_outputs[name + 'classes'] = classes
        processed_outputs[name + 'scores'] = scores

      else:
        raise NotImplementedError('Task type %s is not implemented.' + \
          'Try renaming the task routine.' %name)

    return processed_outputs
  
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
    cmap = get_colormap(cmap_type='cityscapes').numpy()
    dataset = run_lib.inference_dataset(image_path_glob=image_path_glob,
                                        output_dir=output_dir,
                                        preprocess_fn=preprocess_fn)

    class_names = run_lib.load_class_names(class_names_paths=class_names_path)
    if len(class_names) != 2:
      raise ValueError('Class name paths found: %s' %class_names + \
        ' , please specify only 2 (cls, yolo).')
    
    for image, img_filename, save_basename in dataset:

      logits = inference_fn(image)
      if len(logits) != 7:
        raise NotImplementedError("Inferences for multitask only implemented for " +\
          "argmax_outputs=True, visualise_outputs=True, class_present_outputs=True.")

      cls_env, seg_mask, seg_visualised, is_classes_present, yolo_boxes, yolo_classes, yolo_scores = logits
      if yolo_classes.dtype == 'float32':
        yolo_classes, yolo_scores = yolo_scores, yolo_classes

      if save_logits_bin:
        run_lib.write_tensor_as_bin(tensor=image, 
                                    output_path=save_basename + '_input')
        run_lib.write_tensor_as_bin(tensor=seg_mask, 
                                    output_path=save_basename + '_mask')
        run_lib.write_tensor_as_bin(tensor=seg_visualised, 
                                    output_path=save_basename + '_visualised_mask')
        run_lib.write_tensor_as_bin(tensor=yolo_boxes, 
                                    output_path=save_basename + '_boxes')
        run_lib.write_tensor_as_bin(tensor=yolo_scores, 
                                    output_path=save_basename + '_scores')
        run_lib.write_tensor_as_bin(tensor=yolo_classes,
                                    output_path=save_basename + '_classes')

      image = tf.image.resize(image, self._input_image_size)
      image = tf.cast(image, tf.uint8)
      yolo_boxes = box_ops.normalize_boxes(yolo_boxes, self._input_image_size)

      output_image = run_lib.draw_bbox(image=run_lib.tensor_to_numpy(image).squeeze(),
                                       bboxes=run_lib.tensor_to_numpy(yolo_boxes),
                                       scores=run_lib.tensor_to_numpy(yolo_scores),
                                       classes=run_lib.tensor_to_numpy(yolo_classes),
                                       num_bboxes=tf.constant(yolo_classes.shape[0]).numpy(),
                                       class_names=class_names[1])
      env_val = run_lib.tensor_to_numpy(cls_env)[0]
      output_image = run_lib.draw_text(image=output_image, 
                                       text_list=[class_names[0][env_val]],
                                       spacing=20)
      
      seg_mask = tf.squeeze(seg_mask).numpy()
      if seg_mask.ndim > 2:
          seg_mask = np.argmax(seg_mask, axis=-1).astype(np.uint8)
      seg_mask = cmap[seg_mask]
      output_image = np.hstack((output_image, seg_mask))
      
      output_image = tf.image.encode_png(output_image)
      tf.io.write_file(save_basename + '.png', output_image)
      print("Visualised %s, saving result at %s" %(img_filename, save_basename + '.png'))
