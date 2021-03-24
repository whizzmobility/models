r"""Vision models run saved model for inference"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.serving import run_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'Saved model directory.')
flags.DEFINE_string('image_path_glob', None, 'Test image directory.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_boolean('visualise', None, 'Flag to visualise mask.')
flags.DEFINE_boolean('stitch_original', None, 'Flag to stitch image at the side.')
flags.DEFINE_boolean('save_logits_bin', None, 'Flag to save logits bin.')


def main(_):

  imported = tf.saved_model.load(FLAGS.saved_model_dir)
  model_fn = imported.signatures['serving_default']

  def inference_fn(image):
    image = tf.cast(image, dtype=tf.int32)
    return model_fn(image)['predicted_masks']
  
  run_lib.run_inference(FLAGS.image_path_glob, 
                        FLAGS.output_dir,
                        inference_fn, 
                        FLAGS.visualise, 
                        FLAGS.stitch_original,
                        FLAGS.save_logits_bin)


if __name__ == '__main__':
  app.run(main)
