"""Test case for YOLO dataloader configuration definition."""
from typing import List, Optional, Union

from absl.testing import parameterized
import dataclasses
import tensorflow as tf

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.modeling import hyperparams
from official.vision.beta.dataloaders import yolo_input, input_reader_factory
from official.vision.beta.configs.yolo import YoloModel, YoloHead, DataConfig


class YoloInputTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('training', True))
  def test_yolo_input(self, is_training, should_output_image=False):

    params = DataConfig(
      input_path='D:/data/whizz_tf/detect_env*',
      output_size=[256, 256],
      global_batch_size=1,
      is_training=is_training,
      max_bbox_per_scale=150,
      is_bbox_in_pixels=False,
      is_xywh=True)

    model_params = YoloModel(
      num_classes=6,
      input_size=[256, 256, 3],
      head=YoloHead(
        anchor_per_scale=3,
        strides=[16, 32, 64],
        anchors=[12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401],
        xy_scale=[1.2, 1.1, 1.05]
      ))
    
    decoder = yolo_input.Decoder(is_bbox_in_pixels=params.is_bbox_in_pixels,
                                 is_xywh=params.is_xywh)
    parser = yolo_input.Parser(
        output_size=params.output_size,
        input_size=model_params.input_size,
        anchor_per_scale=model_params.head.anchor_per_scale,
        num_classes=model_params.num_classes,
        max_bbox_per_scale=params.max_bbox_per_scale,
        strides=model_params.head.strides,
        anchors=model_params.head.anchors,
        aug_policy=params.aug_policy,
        randaug_magnitude=params.randaug_magnitude,
        randaug_available_ops=params.randaug_available_ops,
        aug_rand_hflip=params.aug_rand_hflip,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        preserve_aspect_ratio=params.preserve_aspect_ratio,
        aug_jitter_im=params.aug_jitter_im,
        aug_jitter_boxes=params.aug_jitter_boxes,
        dtype=params.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=None).take(1)
    
    sample = tf.data.experimental.get_single_element(dataset)
    image, target = sample
    
    if should_output_image:
      output_image = tf.image.convert_image_dtype(tf.squeeze(image_data[0]), dtype=tf.uint8)
      output_image = tf.image.encode_png(output_image)
      tf.io.write_file('D:/Desktop/test.png', output_image)

    self.assertAllEqual(image.shape, (1, 256, 256, 3))

    for i in range(len(target['labels'])):
      self.assertAllEqual(target['labels'][i].shape[-2:], (3,11))
      
      self.assertTrue(tf.reduce_all(target['labels'][i] >= 0))
      self.assertTrue(tf.reduce_all(target['labels'][i][:, :, :, :, :4] <= 256))
      self.assertTrue(tf.reduce_all(target['labels'][i][:, :, :, :, 4:] <= 1))
      
      print('target boxes for %s has >1 %s' %(i, tf.math.reduce_any(target['labels'][i][:, :, :, :, :4] > 1)))

      self.assertAllEqual(target['bboxes'][i].shape[-2:], (150, 4))
      self.assertTrue(tf.reduce_all(tf.math.logical_and(
        target['bboxes'][i] >= 0,
        target['bboxes'][i] <= 256
      )))


if __name__ == '__main__':
  tf.test.main()