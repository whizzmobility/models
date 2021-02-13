r"""Vision models export binary for serving/inference.

To export saved model format into tflite format (shell script):

SAVED_MODEL_DIR = XX
EXPORT_PATH = XX
export_saved_model --saved_model_dir=${SAVED_MODEL_DIR} \
                   --export_path=${EXPORT_PATH}

Run using the same tensorflow version as edge device to ensure ops compatibility
"""

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_dir', None, 'Saved model directory.')
flags.DEFINE_string('export_path', None, 'Export directory.')
flags.DEFINE_string('optimise', None, 'Perform optimisations on model for inference. "size", "float16" or None')
flags.DEFINE_boolean('allow_tf_ops', False, 'Allow tensorflow ops. Increases model size.')


def main(_):

  converter = tf.lite.TFLiteConverter.from_saved_model(
    FLAGS.saved_model_dir, signature_keys=['serving_default'])

  converter.experimental_new_converter = True
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS] # ops builtin optimised for tflite
  
  if FLAGS.allow_tf_ops:
    converter.target_spec.supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS] # for ops not builtin
  if FLAGS.optimise == 'size':
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  if FLAGS.optimise == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
  
  tflite_quant_model = converter.convert()

  with open(FLAGS.export_path, 'wb') as f:
    f.write(tflite_quant_model)


if __name__ == '__main__':
  app.run(main)
