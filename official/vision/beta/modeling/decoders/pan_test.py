"""Tests for PANetDecoder."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import hardnet
from official.vision.beta.modeling.decoders import pan


class PANetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (256, 3),
      (384, 3),
      (512, 3),
      (256, 2)
  )
  def test_network_creation(self, input_size, levels):
    """Test creation of HardnetDecoder."""
    tf.keras.backend.set_image_data_format('channels_last')

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    
    backbone = hardnet.HardNet(model_id=70)
    decoder = pan.PAN(
        routes=levels,
        input_specs=backbone.output_specs)

    endpoints = backbone(inputs)
    feats = decoder(endpoints)
    feats = list(feats.values())

    channels = pan.PANET_SPECS[levels][0]
    size = input_size // 2**(6 - levels + 1) # hardnet downsamples 6x

    for i in range(len(feats)):
      self.assertAllEqual(
        [1, size, size, channels],
        feats[i].shape.as_list()
      )
      channels *= 2
      size /= 2

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        input_specs=hardnet.HardNet(model_id=70).output_specs,
        routes=3,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = pan.PAN(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = pan.PAN.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
