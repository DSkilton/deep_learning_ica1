import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug
        self.score_dense = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, inputs, mask=None):
        scores = self.score_dense(inputs)

        if self.debug:
            tf.print("inputs shape:", tf.shape(inputs))
            tf.print("scores shape:", tf.shape(scores))
            if mask is not None:
                tf.print("mask shape:", tf.shape(mask))

        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)
            scores = scores + (1.0 - mask) * (-1e9)

        weights = tf.nn.softmax(scores, axis=1)

        if self.debug:
            tf.print("weights shape:", tf.shape(weights))

        context = tf.reduce_sum(inputs * weights, axis=1)

        if self.debug:
            tf.print("context shape:", tf.shape(context))

        return context