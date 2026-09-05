#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import warnings
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
import tensorflow as tf
import keras.layers as layers
from keras.layers import Dense, Conv1D

class GPT3Attention(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.d_model = inner_dimension
        self.ffn_dim = inner_dimension * 4  # <--- 鑷姩鎵╁ぇ 4 鍊?
        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        # self.attention = \
        #     layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
        #         if attn_implementation == MultiHeadAttentionImplementation.Keras else\
        #         GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else \
                GPT3Attention(num_heads, self.d_model, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        # [FIX] FFN 涓棿灞備娇鐢ㄦ墿澶у悗鐨?ffn_dim (512)
        self.feed_forward_0 = Conv1D(filters=self.ffn_dim, kernel_size=1, activation="relu",
                                     name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(self.ffn_dim, activation="relu", name=f"{prefix}feed_forward_0")

        # FFN 杈撳嚭灞傚洖鍒?input_dimension (閫氬父绛変簬 d_model)
        self.feed_forward_1 = Conv1D(filters=input_dimension, kernel_size=1, activation="relu",
                                     name=f"{prefix}feed_forward_1") \
            if use_conv else Dense(input_dimension, name=f"{prefix}feed_forward_1")

        # self.feed_forward_0 = Conv1D(filters=inner_dimension, kernel_size=1, activation="relu", name=f"{prefix}feed_forward_0") \
        #     if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        # self.feed_forward_1 = Conv1D(filters=input_dimension, kernel_size=1, activation="relu", name=f"{prefix}feed_forward_1") \
        #     if use_conv else Dense(input_dimension, name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")

    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        # 娉ㄦ剰鍔涘瓙灞傚鐞嗭紙涓嶅彉锛?        x = inputs + attention_output
        x = self.attention_layer_norm(x)  # x 鍗充负娉ㄦ剰鍔涘瓙灞傜殑杈撳嚭 x1

        # 鍓嶉缃戠粶澶勭悊
        ffn_x = self.feed_forward_0(x)
        ffn_x = self.feed_forward_1(ffn_x)
        ffn_x = self.feed_forward_dropout(ffn_x, training=training)  # ffn_out

        # 姝ｇ‘锛氬墠棣堢綉缁滄畫宸繛鎺ワ紙x1 + ffn_out锛夛紝鍐嶅眰褰掍竴鍖?        return self.feed_forward_layer_norm(x + ffn_x)


