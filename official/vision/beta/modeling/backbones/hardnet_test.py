"""Tests for HardNet."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import hardnet


class HardNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(256, 384, 512)
  def test_network_creation(self, input_size):
    """Test creation of HardNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = hardnet.HardNet(model_id=70)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([1, input_size / 2**2, input_size / 2**2, 48],
                        endpoints['0'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**3, input_size / 2**3, 78],
                        endpoints['1'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**4, input_size / 2**4, 160],
                        endpoints['2'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**5, input_size / 2**5, 214],
                        endpoints['3'].shape.as_list())
    self.assertAllEqual([1, input_size / 2**6, input_size / 2**6, 286],
                        endpoints['4'].shape.as_list())

  @parameterized.parameters(1, 3)
  def test_input_specs(self, input_dim):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = hardnet.HardNet(model_id=70, input_specs=input_specs)

    inputs = tf.keras.Input(shape=(256, 256, input_dim), batch_size=1)
    _ = network(inputs)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=70,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None
    )
    network = hardnet.HardNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = hardnet.HardNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
