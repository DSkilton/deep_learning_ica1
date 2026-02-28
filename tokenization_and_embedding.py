import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/keras/Layer
@tf.keras.utils.register_keras_serializable(package="Custom")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, X):
        sequence_length = tf.shape(X)[-1]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = self.pos_emb(positions)
        X = self.token_emb(X)
        return X + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        self.pos_emb.build((input_shape[0], input_shape[1]))
        super().build(input_shape)