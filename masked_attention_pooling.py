import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskedAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, pad_id, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = pad_id
        self.score_dense = tf.keras.layers.Dense(1, activation="tanh")
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # inputs: (batch, seq_len, features)
        # mask:   (batch, seq_len)

        scores = self.score_dense(inputs)
        scores = tf.squeeze(scores, axis=-1)

        if mask is not None:
            minus_inf = tf.cast(-1e9, scores.dtype)
            scores = tf.where(mask, scores, minus_inf)

        weights = tf.nn.softmax(scores, axis=1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted = inputs * weights
        return tf.reduce_sum(weighted, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "pad_id": self.pad_id
        })
        return config