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
flags.DEFINE_string('optimise', None, 'Perform optimisations on model for inference. "size", "float16", "int8" or None')
flags.DEFINE_string('data_file_pattern', None, 'Tfrecord file pattern for representative dataset for int8 quantization.')
flags.DEFINE_boolean('allow_tf_ops', False, 'Allow tensorflow ops. Increases model size.')


def main(_):

  converter = tf.lite.TFLiteConverter.from_saved_model(
    FLAGS.saved_model_dir, signature_keys=['serving_default'])

  converter.experimental_new_converter = True
  converter.experimental_new_quantizer = True
  
  if FLAGS.allow_tf_ops:
    converter.target_spec.supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS] # for ops not builtin
  if FLAGS.optimise == 'size':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  if FLAGS.optimise == 'float16':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if FLAGS.optimise == 'int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    def representative_data_gen():
      model = tf.saved_model.load(FLAGS.saved_model_dir)
      concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      input_shape = concrete_func.inputs[0].shape[1:-1]

      matched_files = tf.io.gfile.glob(FLAGS.data_file_pattern)
      keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value='')
      }
      dataset = tf.data.Dataset.from_tensor_slices(matched_files)
      dataset = dataset.interleave(map_func=tf.data.TFRecordDataset)
      
      for data in dataset.batch(1).take(100):
        decoded_data = tf.io.parse_single_example(data[0], keys_to_features)
        image = tf.io.decode_image(decoded_data['image/encoded'], channels=3)
        image = tf.image.resize(image, input_shape, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.uint8)
        yield [image]

    converter.representative_dataset = representative_data_gen
  
  tflite_quant_model = converter.convert()

  with open(FLAGS.export_path, 'wb') as f:
    f.write(tflite_quant_model)


if __name__ == '__main__':
  app.run(main)
