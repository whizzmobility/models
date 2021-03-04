"""
Loads pretrained checkpoint and syncs name according to given tf.keras.Model specified
Requires TF1.x support, since variables are a TF1 thing.
"""

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt_ref', None, 'Checkpoint to reference shape against and copy name from.')
flags.DEFINE_string('ckpt_target', None, 'Target checkpoint which names will be changed.')

def main(_):

  # model_shapes = [list(weight.shape) for weight in model.get_weights()]
  # model_names = [weight.name for layer in model.layers for weight in layer.weights]
  # assert len(model_shapes) == len(model_names)
  # for m in model_shapes:
  #   print(m)

  ref_vars = [i for i in tf.train.list_variables(FLAGS.ckpt_ref)]
  for name, shape in ref_vars:
    print(name, shape)
  
  tar_vars = [i for i in tf.train.list_variables(FLAGS.ckpt_target)]
  for name, shape in tar_vars:
    print(name, shape)
  
  # matches = []
  # i = 0
  # for ckpt_name, ckpt_shape in checkpoint_vars:
  #   print(ckpt_shape, model_shapes[i])

  #   if ckpt_shape == model_shapes[i]:
  #     matches.append((ckpt_name, model_names[i], ckpt_shape, model_shapes[i]))
  #     i += 1
  
  # for m in matches:
  #   print(m)

  # with tf.compat.v1.Session() as sess:

  #   for var_name, _ in tf.train.list_variables(FLAGS.ckpt_ref)[1:]:
  #     # Load the variable
  #     var = tf.train.load_variable(FLAGS.ckpt_ref, var_name)

  #     # Set the new name
  #     new_name = var_name
  #     new_name = new_name.replace('layer', 'haha')

  #     print('%-50s ==> %-50s' % (var_name, new_name))
  #     # Rename the variable
  #     var = tf.Variable(var, name=new_name)

  #   # Save the variables
  #   saver = tf.compat.v1.train.Saver()
  #   sess.run(tf.compat.v1.global_variables_initializer())
  #   saver.save(sess, './output_models/model.ckpt')
  
  a = 1/0

if __name__ == '__main__':
  app.run(main)
