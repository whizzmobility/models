r"""Vision models run tflite model for inference"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.serving import run_lib
from official.common import flags as tfm_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'Tflite model path.')


def main(_):

  export_module = run_lib.get_export_module(experiment=FLAGS.experiment,
                                            batch_size=FLAGS.batch_size)

  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  def inference_fn(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    outputs = []
    for i in range(len(output_details)):
      outputs.append(interpreter.get_tensor(output_details[i]['index']))
    return outputs
  
  def preprocess_fn(image):
    image = tf.image.resize(image, input_details[0]['shape'][1:3])
    image = tf.cast(image, dtype=tf.uint8)
    return image

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
