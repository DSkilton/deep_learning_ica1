import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class StripMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        return None