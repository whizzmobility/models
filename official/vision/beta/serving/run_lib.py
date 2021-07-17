r"""Vision models run inference utility function."""

from typing import Callable, Optional, List, Mapping
import os
import glob
import json
import random
import colorsys

from absl import flags
import cv2
import numpy as np
import tensorflow as tf

from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta import configs
from official.modeling.multitask import configs as multi_cfg
from official.vision.beta.serving import detection
from official.vision.beta.serving import image_classification
from official.vision.beta.serving import semantic_segmentation
from official.vision.beta.serving import yolo
from official.vision.beta.serving import multitask

IMAGENET_MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def define_flags():
  """Defines flags specific for running inferences."""
  # inference model flags
  flags.DEFINE_integer('batch_size', 1, 'The batch size.')

  # inference data flags
  flags.DEFINE_string('image_path_glob', None, 'Test image directory.')
  flags.DEFINE_string('output_dir', None, 'Output directory.')
  flags.DEFINE_boolean('save_logits_bin', None, 'Flag to save logits bin.')

  # optional flags, supplied as kwargs
  flags.DEFINE_boolean('visualise', None, '(Segmentation) Flag to visualise mask.')
  flags.DEFINE_boolean('stitch_original', None, '(Segmentation) Flag to stitch image at the side.')
  flags.DEFINE_string('class_names_path', None, '(Yolo/Cls) Csv of paths to txt' + \
    ' files with class names.')


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
                      batch_size: int,
                      config_files: Optional[str] = None):
  """Get export module according to experiment config.
  
  Args:
    experiment: `str`, look up for ExperimentConfig factory methods
    batch_size: `int`, batch size of inference
    config_file: `str`, path to yaml file that overrides experiment config.
  """
  params = exp_factory.get_exp_config(experiment)
  for config_file in config_files or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)
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


def read_class_names(class_names_path):
  """Reads class names from text file.
  Supports .txt and .json.
  
  Args:
    class_names_path: `str`, path to json/txt file containing classes. 
      Text file should contain one class name per line.
      Json file should contain only one dictionary, `Mapping[int, str]`
  """
  
  names = {}
  if class_names_path.endswith('.txt'):
    with open(class_names_path, 'r') as data:
      for idx, name in enumerate(data):
        names[idx] = name.strip('\n')
  
  elif class_names_path.endswith('.json'):
    with open(class_names_path) as f:
      names = json.load(f)
    if type(list(names.keys())[0]) == str and type(list(names.values())[0]) == int:
      names = dict((v,k) for k,v in names.items())

  else:
    raise NotImplementedError('File type is not .txt or .json, path %s' %class_names_path)

  assert type(list(names.keys())[0]) == int, 'Loaded dict %s has wrong key type %s' %(
    class_names_path, type(list(names.keys()[0])))
  assert type(list(names.values())[0]) == str, 'Loaded dict %s has wrong value type %s' %(
    class_names_path, type(list(names.values()[0])))

  return names


def draw_bbox(image: np.array, 
              bboxes: np.array,
              scores: np.array,
              classes: np.array,
              num_bboxes: np.array,
              class_names: Mapping[int, str], 
              show_label: bool = True):
  num_classes = len(class_names)
  image_h, image_w, _ = image.shape
  hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

  random.seed(0)
  random.shuffle(colors)
  random.seed(None)

  for i in range(num_bboxes[0]):
    if int(classes[0][i]) < 0 or int(classes[0][i]) > num_classes: continue
    coor = bboxes[0][i] * [image_h, image_w, image_h, image_w]
    coor = coor.astype(np.int32)

    fontScale = 0.5
    score = scores[0][i]
    class_ind = int(classes[0][i])
    bbox_color = colors[class_ind]
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
    cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

    if show_label:
      bbox_mess = '%s: %.2f' % (class_names[class_ind], score)
      t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
      c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
      cv2.rectangle(image, c1, c3, bbox_color, -1) #filled

      cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
  return image


def draw_text(image: np.array, 
              text_list: List[str],
              spacing: int = 50):
    """ Writes list of messages on image, separated by newline 
    
    Args:
      image: `np.array`, of shape (height, width, 3), RGB image
      text_list: `List[str]`, list of texts to write on each line
      spacing: `int`, spacing in pixels
    """

    pos = spacing
    for text in text_list:
        image = cv2.putText(image, text,
                            (spacing, pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        pos += spacing

    return image
