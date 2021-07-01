"""Convert jpg/png images and png masks to TFRecord format."""

import logging
import os
import glob

from absl import app  # pylint:disable=unused-import
from absl import flags
import cv2
import tensorflow as tf

from official.vision.beta.data import tfrecord_lib

flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('seg_dir', '', 'Directory containing images.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def generate_annotations(images_filenames, seg_filenames):
    
    for image_filename, seg_filename in zip(images_filenames, seg_filenames):
        
        image_dir = os.path.dirname(image_filename)
        image = cv2.imread(image_filename)
        image_size = image.shape
        image_data = {
            'height': image_size[0],
            'width': image_size[1],
            'file_name': image_filename,
            'id': image_filename
        }

        seg_dir = os.path.dirname(seg_filename)
        seg_data = {
            'height': image_size[0],
            'width': image_size[1],
            'file_name': seg_filename,
            'id': seg_filename
        }

        yield (image_data, image_dir, seg_data, seg_dir)


def create_tf_example(image,
                      image_dir,
                      seg,
                      seg_dir):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
        image_dir: directory containing the image files.
        seg: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
        seg_dir: directory containing the image files.

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
    
    seg_full_path = os.path.join(seg_dir, seg['file_name'])
    with tf.io.gfile.GFile(seg_full_path, 'rb') as fid:
        seg_encoded_img = fid.read()
    feature_dict['image/segmentation/class/encoded'] = tfrecord_lib.convert_to_feature(seg_encoded_img)
    
    num_annotations_skipped = 0 # data checks

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped


def _create_tf_record_from_imgs(image_dir,
                                seg_dir,
                                output_path,
                                num_shards):
    """ Loads images and segmentation masks and converst to tf.Record format 
    
    Args:
        image_dir: Directory containing the image files.
        seg_dir: Directory containing the segmentation files.
        output_path: Path to output tf.Record file.
        num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)
    img_filenames = tfrecord_lib.get_all_files(image_dir, extension=[".png", ".jpg"])
    seg_filenames = tfrecord_lib.get_all_files(seg_dir, extension=[".png", ".jpg"])
    assert len(img_filenames) == len(seg_filenames), "Found %s images but %s ground truth masks." %(
        len(img_filenames), len(seg_filenames))
    logging.info("Found total of %s images and ground truth masks." %len(img_filenames))

    coco_annotations_iter = generate_annotations(img_filenames, seg_filenames)

    """
    create_tf_example/feature_dict, see tfrecord_lib.image_info_to_feature_dict
    Decoder (parses single example proto)
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=0),
    'image/segmentation/class/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')

    Parser (parse image and annotations into dictionary of tensors)
    """

    # annotation_iterator: An iterator of tuples containing details about the
    #   dataset.
    # process_func: A function which takes the elements from the tuples of
    #   annotation_iterator as arguments and returns a tuple of (tf.train.Example,
    #   int). The integer indicates the number of annotations that were skipped.
    num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, coco_annotations_iter, create_tf_example, num_shards)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
    assert FLAGS.image_dir, '`image_dir` missing.'
    assert FLAGS.seg_dir, '`seg_dir` missing.'
    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)
    
    _create_tf_record_from_imgs(FLAGS.image_dir,
                                FLAGS.seg_dir,
                                FLAGS.output_file_prefix,
                                FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)