"""Tests for HardNet."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import hardnet
from official.vision.beta.modeling.decoders import hardnet_decoder


class HardnetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (70, 256),
      (70, 384),
      (70, 512)
  )
  def test_network_creation(self, model_id, input_size):
    """Test creation of HardnetDecoder."""
    tf.keras.backend.set_image_data_format('channels_last')

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    backbone = hardnet.HardNet(model_id=70)
    network = hardnet_decoder.HardNetDecoder(
        model_id=model_id,
        input_specs=backbone.output_specs)

    endpoints = backbone(inputs)
    feats = network(endpoints)
    feats = list(feats.values())[0] # only one output

    # number of stride downs at stem = 2, size of last downsampling channel at stem = 48
    self.assertAllEqual(
        [1, input_size // 2**2, input_size // 2**2, 48],
        feats.shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        input_specs=hardnet.HardNet(model_id=70).output_specs,
        model_id=70,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = hardnet_decoder.HardNetDecoder(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = hardnet_decoder.HardNetDecoder.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
