"""Tests for classification_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.heads import classification_heads


class ClassificationHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (3, 3, True, 0.2),
      (3, 0, True, 0.2),
      (3, 3, False, 0.2),
      (3, 3, False, 0),
  )
  def test_forward(self, 
                   level, 
                   num_convs, 
                   add_head_batch_norm, 
                   dropout_rate):
    head = classification_heads.ClassificationHead(
        num_classes=10, level=level, num_convs=num_convs,
        add_head_batch_norm=add_head_batch_norm, dropout_rate=dropout_rate)
    backbone_features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    decoder_features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    logits = head(backbone_features, decoder_features)

    if level in decoder_features:
      self.assertAllEqual(logits.numpy().shape, [
          2, 10
      ])

  def test_serialize_deserialize(self):
    head = classification_heads.ClassificationHead(num_classes=10, level=3)
    config = head.get_config()
    new_head = classification_heads.ClassificationHead.from_config(config)
    self.assertAllEqual(head.get_config(), new_head.get_config())

if __name__ == '__main__':
  tf.test.main()
