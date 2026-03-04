import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = feed_forward_dim
        self.rate = rate
        
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name="multihead_attention")
        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])

        # One for attnention output and one for feed forward output
        self.layer_normalization_att = tf.keras.layers.LayerNormalization(epsilon=1e-6) # prevents division by zero
        self.layer_normalization_ff= tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_att = tf.keras.layers.Dropout(rate)
        self.dropout_ff = tf.keras.layers.Dropout(rate)

        self.last_attention_scores = None

    def call(self, inputs, training=False, tensor_mask=None, return_attention_scores=False, return_debug=False):
        if tensor_mask is not None:
            tensor_mask = tf.cast(tensor_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)

        if return_attention_scores:
            attention_output, attention_scores = self.attention(
                inputs, 
                inputs, 
                attention_mask=tensor_mask, 
                return_attention_scores=True,
            )
            self.last_attention_scores = attention_scores
        else: 
            attention_output = self.attention(
                inputs, 
                inputs, 
                attention_mask=tensor_mask
            )
            attention_scores = None
            self.last_attention_scores = None

        attention_output = self.dropout_att(attention_output, training=training)
        output = self.layer_normalization_att(inputs + attention_output)

        feed_forward_output = self.feed_forward_network(output)
        feed_forward_output = self.dropout_ff(feed_forward_output, training=training)

        X = self.layer_normalization_ff(output + feed_forward_output)

        if return_debug:
            return X, attention_scores, output, X
            
        if return_attention_scores:
            return X, attention_scores
            
        return X

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config