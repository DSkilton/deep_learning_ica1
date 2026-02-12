import tensorflow as tf 
from tensorflow import layers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        
        # One for attnention output and one for feed forward output
        self.layer_normalization_att = tf.keras.layers.LayerNormalization(epsilon=1e-6) # prevents division by zero
        self.layer_normalization_ff= tf.keras.layers.LayerNormalization(epsilon=1e-6) 
        self.dropout_att = layers.Dropout(rate)
        self.dropout_ff = layers.Dropout(rate)
        
    def call(self, inputs, training=None, mask=None):
        attention_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
        
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        attention_output = self.dropout_att(attention_output, training=training)
        output = self.layer_normalization_att(inputs + attention_output)
        
        feed_forward_output = self.feed_forward_network(output)
        feed_forward_output = self.dropout_ff(feed_forward_output, training=training)
        
        return self.layer_normalization_ff(output + feed_forward_output)