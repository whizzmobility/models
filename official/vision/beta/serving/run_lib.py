r"""Vision models run inference utility function."""

from typing import Callable
import os
import glob

import tensorflow as tf

# from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta import configs
from official.modeling.multitask import configs as multi_cfg
from official.vision.beta.serving import detection
from official.vision.beta.serving import image_classification
from official.vision.beta.serving import semantic_segmentation
from official.vision.beta.serving import yolo
from official.vision.beta.serving import multitask


def inference_dataset(image_path_glob: str,
                      output_dir: str,
                      preprocess_fn: Callable[[tf.Tensor], tf.Tensor] = None):
  """Creates generator of image tensors from image path glob
  
  Args:
    image_path_glob: `str`, path pattern for images
    output_dir: `str`, path to output logs
    preprocess_fn: `Callable`, takes image tensor of shape (1, height, 
      width, channels), produces altered image tensor of same shape

  Yields:
    `image`: `tf.Tensor`, image of shape (1, height, width, channels)
    `img_filename`: `str`, path to image instance
    `save_basename`: `str`, base save path for logs or outputs
  """
  img_filenames = [f for f in glob.glob(image_path_glob, recursive=True)]
  image_dir = image_path_glob.split("*")[0].strip(os.sep).strip('/')

  for img_filename in img_filenames:
    image = tf.io.read_file(img_filename)
    image_format = os.path.splitext(img_filename)[-1]
    if image_format == ".png":
        image = tf.image.decode_png(image)
    elif image_format == ".jpg":
        image = tf.image.decode_jpeg(image)
    else:
        raise NotImplementedError("Unable to decode %s file type." %(image_format))
    
    image = tf.expand_dims(image, axis=0)
    if preprocess_fn:
      image = preprocess_fn(image)

    # generate save path
    save_basename = img_filename.replace(image_dir, output_dir)
    save_basename = os.path.splitext(save_basename)[0]
    save_dir = os.path.dirname(save_basename)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    yield image, img_filename, save_basename


def write_tensor_as_bin(tensor: tf.Tensor,
                        output_path: str):
  """Write tensor as a binary file.
  Uses big endian for compatibility with whizzscooters/raven-android tests.
  """
  if tensor.dtype == tf.float32:
    tensor.numpy().flatten().astype(">f4").tofile(output_path)
  elif tensor.dtype == tf.int32:
    tensor.numpy().flatten().astype(">i4").tofile(output_path)
  elif tensor.dtype == tf.uint8:
    tensor.numpy().flatten().astype(">i1").tofile(output_path)
  else:
    raise NotImplementedError('Saving for %s is not implemented.' %(
      tensor.dtype))


def get_export_module(experiment: str,
                      batch_size: int):
  """Get export module according to experiment config.
  
  Args:
    experiment: `str`, look up for ExperimentConfig factory methods
    batch_size: `int`, batch size of inference
  """
  params = exp_factory.get_exp_config(experiment)
  params.validate()
  params.lock()

  # Obtain relevant serving object
  kwargs = dict(params=params,
                batch_size=batch_size,
                input_image_size=params.task.model.input_size[:2],
                num_channels=3)

  if isinstance(params.task,configs.image_classification.ImageClassificationTask):
    export_module = image_classification.ClassificationModule(**kwargs)
  elif isinstance(params.task, configs.retinanet.RetinaNetTask) or \
    isinstance(params.task, configs.maskrcnn.MaskRCNNTask):
    export_module = detection.DetectionModule(**kwargs)
  elif isinstance(params.task, configs.semantic_segmentation.SemanticSegmentationTask):
    export_module = semantic_segmentation.SegmentationModule(**kwargs)
  elif isinstance(params.task, configs.yolo.YoloTask):
    export_module = yolo.YoloModule(**kwargs)
  elif isinstance(params.task, multi_cfg.MultiTaskConfig):
    export_module = multitask.MultitaskModule(**kwargs)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))

  return export_module
