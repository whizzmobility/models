r"""Vision models run saved model for inference"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.serving import run_lib
from official.common import flags as tfm_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'Saved model directory.')


def main(_):

  export_module = run_lib.get_export_module(experiment=FLAGS.experiment,
                                            batch_size=FLAGS.batch_size)

  imported = tf.saved_model.load(FLAGS.saved_model_dir)
  model_fn = imported.signatures['serving_default']

  def preprocess_fn(image):
    image = tf.image.resize(image, export_module._input_image_size)
    image = tf.cast(image, dtype=tf.uint8)
    return image

  def inference_fn(image):
    return [v for k, v in model_fn(image).items()]
  
  export_module.run(image_path_glob=FLAGS.image_path_glob, 
                    output_dir=FLAGS.output_dir,
                    preprocess_fn=preprocess_fn,
                    inference_fn=inference_fn, 
                    visualise=FLAGS.visualise, 
                    stitch_original=FLAGS.stitch_original,
                    class_names_path=FLAGS.class_names_path,
                    save_logits_bin=FLAGS.save_logits_bin)


if __name__ == '__main__':
  tfm_flags.define_flags()
  run_lib.define_flags()
  app.run(main)
