r"""Vision models run saved model for inference"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.serving import run_lib
from official.common import flags as tfm_flags

FLAGS = flags.FLAGS

# inference model flags
flags.DEFINE_integer('batch_size', None, 'The batch size.')
flags.DEFINE_string('saved_model_dir', None, 'Saved model directory.')

# inference data flags
flags.DEFINE_string('image_path_glob', None, 'Test image directory.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_boolean('save_logits_bin', None, 'Flag to save logits bin.')

# unique flags
flags.DEFINE_boolean('visualise', None, 'Flag to visualise mask.')
flags.DEFINE_boolean('stitch_original', None, 'Flag to stitch image at the side.')


def main(_):

  export_module = run_lib.get_export_module(experiment=FLAGS.experiment,
                                            batch_size=FLAGS.batch_size)

  imported = tf.saved_model.load(FLAGS.saved_model_dir)
  model_fn = imported.signatures['serving_default']

  def inference_fn(image):
    return [v.numpy() for k, v in model_fn(image).items()]
  
  export_module.run(image_path_glob=FLAGS.image_path_glob, 
                    output_dir=FLAGS.output_dir,
                    preprocess_fn=None,
                    inference_fn=inference_fn, 
                    visualise=FLAGS.visualise, 
                    stitch_original=FLAGS.stitch_original,
                    save_logits_bin=FLAGS.save_logits_bin)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
