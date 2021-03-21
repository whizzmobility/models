r"""Vision models run inference using trained checkpoint"""

from absl import app
from absl import flags
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.serving import run_lib

from official.common import flags as tfm_flags
from official.vision.beta import configs
from official.vision.beta.serving import detection
from official.vision.beta.serving import image_classification
from official.vision.beta.serving import semantic_segmentation

FLAGS = flags.FLAGS

flags.DEFINE_string('image_path_glob', None, 'Test image directory.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_boolean('visualise', None, 'Flag to visualise mask.')
flags.DEFINE_boolean('stitch_original', None, 'Flag to stitch image at the side.')
flags.DEFINE_integer('batch_size', None, 'The batch size.')
flags.DEFINE_string(
    'input_image_size', '224,224',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


def main(_):
  
  params = exp_factory.get_exp_config(FLAGS.experiment)
  for config_file in FLAGS.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)
  if FLAGS.params_override:
    params = hyperparams.override_params_dict(
        params, FLAGS.params_override, is_strict=True)

  params.validate()
  params.lock()

  # Obtain relevant serving object
  input_image_size = [int(x) for x in FLAGS.input_image_size.split(',')]
  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    export_module = image_classification.ClassificationModule(
        params=params, batch_size=FLAGS.batch_size, input_image_size=input_image_size)
  elif isinstance(params.task, configs.retinanet.RetinaNetTask) or isinstance(
      params.task, configs.maskrcnn.MaskRCNNTask):
    export_module = detection.DetectionModule(
        params=params, batch_size=FLAGS.batch_size, input_image_size=input_image_size)
  elif isinstance(params.task,
                  configs.semantic_segmentation.SemanticSegmentationTask):
    export_module = semantic_segmentation.SegmentationModule(
        params=params, batch_size=FLAGS.batch_size, input_image_size=input_image_size)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))
  
  model = export_module._build_model()

  ckpt = tf.train.Checkpoint(model=model)

  ckpt_dir_or_file = FLAGS.model_dir
  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  status = ckpt.restore(ckpt_dir_or_file).expect_partial()

  def inference_fn(images):
    return export_module.serve(images)['predicted_masks']
  
  run_lib.run_inference(FLAGS.image_path_glob,
                        FLAGS.output_dir,
                        inference_fn,
                        FLAGS.visualise,
                        FLAGS.stitch_original)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
