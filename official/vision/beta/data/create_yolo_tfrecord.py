"""Convert jpg/png images and corresponding yolo txts to TFRecord format.
Converts into COCO similar format since existing parser works for COCO.
"""

import logging
import os
import glob
import json
from typing import Hashable, Dict, Tuple

from absl import app  # pylint:disable=unused-import
from absl import flags
import cv2
import tensorflow as tf

from official.vision.beta.data import tfrecord_lib

flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def get_text_filename(image_filename):
  possible_filenames = [f for f in glob.glob(
    os.path.splitext(image_filename)[0] + '*.txt'
  )]

  if len(possible_filenames) > 1:
    raise ValueError(f'Found {possible_filenames} candidate ' + \
      'yolo txt files for {image_filename}')
  elif len(possible_filenames) == 1:
    return possible_filenames[0]
  else:
    return None


def generate_annotations(images_filenames: str, 
                         *args) -> Tuple[str, str]:
    """
    Returns iterable of tuples containing details of dataset to be passed to
    create_tf_example.
    """

    for image_filename in images_filenames:
        
        dir = os.path.dirname(image_filename)
        text_filename = get_text_filename(image_filename)
        if text_filename is None:
          continue

        image = cv2.imread(image_filename)
        image_size = image.shape

        image_data = {
            'height': image_size[0],
            'width': image_size[1],
            'file_name': image_filename,
            'id': image_filename
        }

        with open(text_filename, 'r') as f:  
          data = f.read().split('\n')
        
        bbox_data = dict((k, list()) for k in ['class', 'x', 'y', 'w', 'h'])
        for line in data:
          if line == '': continue
          box_class, x, y, w, h = line.split(' ')
          bbox_data['class'].append(float(box_class))
          bbox_data['x'].append(float(x))
          bbox_data['y'].append(float(y))
          bbox_data['w'].append(float(w))
          bbox_data['h'].append(float(h))

        yield (image_data, bbox_data, dir, *args)


def create_tf_example(image: Dict,
                      bbox: Dict,
                      image_dir: str) -> Tuple[tf.train.Example, int]:
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
        bbox: dict with keys: [u'x', u'y', u'width', u'height']
        image_dir: directory containing the image files.

    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    filename = image['file_name']
    img_format = os.path.splitext(filename)[-1]

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_img = fid.read()

    feature_dict = tfrecord_lib.image_info_to_feature_dict(
        image['height'], image['width'], filename, image['id'], encoded_img, img_format)
    feature_dict['bbox/class'] = tfrecord_lib.convert_to_feature(bbox['class'])
    feature_dict['bbox/x'] = tfrecord_lib.convert_to_feature(bbox['x'])
    feature_dict['bbox/y'] = tfrecord_lib.convert_to_feature(bbox['y'])
    feature_dict['bbox/w'] = tfrecord_lib.convert_to_feature(bbox['w'])
    feature_dict['bbox/h'] = tfrecord_lib.convert_to_feature(bbox['h'])

    num_annotations_skipped = 0 # data checks

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped


def _create_tf_record_from_imgs(image_dir: str,
                                output_path: str,
                                num_shards: int) -> None:
    """ Loads images and segmentation masks and converst to tf.Record format 
    
    Args:
        image_dir: Directory containing the image files.
        output_path: Path to output tf.Record file.
        num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)

    img_filenames = tfrecord_lib.get_all_files(image_dir, extension=[".png", ".jpg"])
    logging.info("Found total of %s images." %len(img_filenames))

    coco_annotations_iter = generate_annotations(img_filenames)

    num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, coco_annotations_iter, create_tf_example, num_shards)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
    assert FLAGS.image_dir, '`image_dir` missing.'
    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)
    
    _create_tf_record_from_imgs(FLAGS.image_dir,
                                FLAGS.output_file_prefix,
                                FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
