import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class TokenMask(tf.keras.layers.Layer):
    def __init__(self, pad_id, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = pad_id

    def call(self, inputs):
        return tf.not_equal(inputs, self.pad_id)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pad_id": self.pad_id
        })
        return config