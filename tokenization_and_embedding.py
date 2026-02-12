import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/keras/Layer
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)    
        
    def call(self, X):
        max_len = tf.shape(X)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        X = self.token_emb(X)
        return X + positions