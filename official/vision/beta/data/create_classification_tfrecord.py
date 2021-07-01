"""Convert jpg/png images and corresponding json label to TFRecord format."""

import logging
import os
import json
from typing import Hashable, Dict, Tuple

from absl import app  # pylint:disable=unused-import
from absl import flags
import cv2
import tensorflow as tf

from official.vision.beta.data import tfrecord_lib

flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('classes_json', '', 'Json mapping class names to class number.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_string('json_key', '', 'Key in json file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def generate_annotations(images_filenames: str, 
                         *args) -> Tuple[str, str]:
    """
    Returns iterable of tuples containing details of dataset to be passed to
    create_tf_example.
    """

    for image_filename in images_filenames:
        
        image_dir = os.path.dirname(image_filename)
        image = cv2.imread(image_filename)
        image_size = image.shape
        image_data = {
            'height': image_size[0],
            'width': image_size[1],
            'file_name': image_filename,
            'id': image_filename
        }

        yield (image_data, image_dir, *args)


def get_json_param_value(img_path:str, 
                         json_param:str) -> Hashable:
    """
    Obtains raw label value from an image's corresponding json file.
    TODO: abstract this out so the script can be used for more general cases
    """

    dirname = os.path.dirname(img_path)
    img_basename = os.path.basename(img_path)
    img_splitname = os.path.splitext(img_basename)[0].split('_')

    json_path = os.path.join(
        dirname,
        'measurements_%05d_%s.json' \
        %(int(img_splitname[1]), img_splitname[2])
    )
    
    with open(json_path, 'r') as measurement_file:
        measurements = json.load(measurement_file)
    
    if json_param not in measurements.keys():
        print("Json param %s is not found in %s." %(json_param, json_path))
    
    return measurements[json_param]


def create_tf_example(image: Dict,
                      image_dir: str,
                      classes_map: Dict,
                      json_key: str) -> Tuple[tf.train.Example, int]:
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
        image_dir: directory containing the image files.
        classes_map: dictionary mapping raw class names to class index
        json_key: key to read from in corresponding json file

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
    
    raw_label = get_json_param_value(full_path, json_key) # int64 for classification decoder
    label = classes_map[raw_label]

    feature_dict['image/class/label'] = tfrecord_lib.convert_to_feature(label)
    
    num_annotations_skipped = 0 # data checks

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped


def _create_tf_record_from_imgs(image_dir: str,
                                output_path: str,
                                classes_json: str,
                                json_key: str,
                                num_shards: int) -> None:
    """ Loads images and segmentation masks and converst to tf.Record format 
    
    Args:
        image_dir: Directory containing the image files.
        classes_json: Path to json file containing mapping for labels
        output_path: Path to output tf.Record file.
        json_key: key to read from in corresponding json file
        num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)
    if not os.path.exists(classes_json):
      raise ValueError('%s cannot be found. Specify valid json path.' %classes_json)
    with open(classes_json) as f:
      classes_map = json.load(f)

    img_filenames = tfrecord_lib.get_all_files(image_dir, extension=[".png", ".jpg"])
    logging.info("Found total of %s images." %len(img_filenames))

    coco_annotations_iter = generate_annotations(img_filenames, classes_map, json_key)

    num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, coco_annotations_iter, create_tf_example, num_shards)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
    assert FLAGS.image_dir, '`image_dir` missing.'
    assert FLAGS.classes_json, '`classes_json` missing.'
    assert FLAGS.json_key, 'specify valid `json_key`.'
    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)
    
    _create_tf_record_from_imgs(FLAGS.image_dir,
                                FLAGS.output_file_prefix,
                                FLAGS.classes_json,
                                FLAGS.json_key,
                                FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
