r"""Vision models run tflite model for inference"""

from absl import app
from absl import flags
import tensorflow as tf
import cv2

from official.vision.beta.serving import run_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'Saved model directory.')
flags.DEFINE_string('image_path_glob', None, 'Test image directory.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_boolean('visualise', None, 'Flag to visualise mask.')
flags.DEFINE_boolean('stitch_original', None, 'Flag to stitch image at the side.')
flags.DEFINE_boolean('save_logits_bin', None, 'Flag to save logits bin.')


def main(_):

  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  input_type = input_details[0]['dtype']
  input_image_dims = input_details[0]['shape'][1:3].tolist()
  output_details = interpreter.get_output_details()
  
  def inference_fn(image):
    image = tf.cast(image, dtype=tf.int32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])
  
  def preprocess_fn(image):
    return tf.image.resize(image, input_details[0]['shape'][1:3])

  run_lib.run_inference(FLAGS.image_path_glob, 
                        FLAGS.output_dir,
                        inference_fn, 
                        FLAGS.visualise, 
                        FLAGS.stitch_original,
                        FLAGS.save_logits_bin,
                        preprocess_fn)


if __name__ == '__main__':
  app.run(main)
