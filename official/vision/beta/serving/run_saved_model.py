r"""Vision models run saved model for inference.

To export a trained checkpoint in saved_model format (shell script):

SAVED_MODEL_DIR = XX
IMAGE_DIR = XX
export_saved_model --saved_model_dir=${SAVED_MODEL_DIR} \
                   --image_dir=${IMAGE_DIR}
"""

import os
import glob

import numpy as np
from absl import app
from absl import flags
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.ops import preprocess_ops

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'Saved model directory.')
flags.DEFINE_string('image_dir_glob', None, 'Test image directory.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_boolean('visualise', None, 'Flag to visualise mask.')
# flags.DEFINE_string(
#     'input_image_size', '224,224',
#     'The comma-separated string of two integers representing the height,width '
#     'of the input to the model.')

CITYSCAPES_COLORMAP = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [255, 255, 255]
], dtype=np.uint8)

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
IMAGE_SIZE = (512, 512)


def build_inputs(image):
    """Builds classification model inputs for serving."""

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    image, _ = preprocess_ops.resize_and_crop_image(
        image,
        IMAGE_SIZE,
        padded_size=IMAGE_SIZE,
        aug_scale_min=1.0,
        aug_scale_max=1.0)

    return image


def main(_):

  imported = tf.saved_model.load(FLAGS.saved_model_dir)
  model_fn = imported.signatures['serving_default']

  img_filenames = [f for f in glob.glob(FLAGS.image_dir_glob, recursive=True)]
  image_dir = FLAGS.image_dir_glob.split("*")[0].strip(os.sep).strip('/')

  for img_filename in img_filenames:
    image = tf.io.read_file(img_filename)
    image_format = os.path.splitext(img_filename)[-1]
    if image_format == ".png":
        image = tf.image.decode_png(image)
    elif image_format == ".jpg":
        image = tf.image.decode_jpeg(image)
    else:
        raise NotImplementedError("Unable to decode %s file type." %(image_format))
    
    # encoded_jpg = tf.image.encode_jpeg(image)
    # tf.io.write_file('img.jpg', encoded_jpg)
    
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)
    logits = model_fn(image)['predicted_masks']
    logits = np.squeeze(logits.numpy())
    seg_map = np.argmax(logits, axis=2).astype(np.uint8)

    if FLAGS.visualise:
      seg_map = CITYSCAPES_COLORMAP[seg_map]
    
    encoded_seg_map = tf.image.encode_png(seg_map)
    save_path = img_filename.replace(image_dir, FLAGS.output_dir)
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    tf.io.write_file(save_path, encoded_seg_map)
    print("Visualised %s, saving result at %s" %(img_filename, save_path))


if __name__ == '__main__':
  app.run(main)
