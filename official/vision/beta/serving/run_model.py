r"""Vision models run inference using trained checkpoint"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.serving import run_lib
from official.common import flags as tfm_flags

FLAGS = flags.FLAGS

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


def main(_):
  
  export_module = run_lib.get_export_module(experiment=FLAGS.experiment,
                                            batch_size=FLAGS.batch_size,
                                            config_files=FLAGS.config_file)
  
  ckpt = tf.train.Checkpoint(model=export_module.model)
  ckpt_dir_or_file = FLAGS.model_dir
  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  status = ckpt.restore(ckpt_dir_or_file).expect_partial()

  def inference_fn(images):
    outputs = export_module.serve(images)
    return [v for k, v in outputs.items()]
  
  export_module.run(image_path_glob=FLAGS.image_path_glob, 
                    output_dir=FLAGS.output_dir,
                    preprocess_fn=None,
                    inference_fn=inference_fn, 
                    visualise=FLAGS.visualise, 
                    stitch_original=FLAGS.stitch_original,
                    class_names_path=FLAGS.class_names_path,
                    save_logits_bin=FLAGS.save_logits_bin)


if __name__ == '__main__':
  tfm_flags.define_flags()
  run_lib.define_flags()
  app.run(main)
