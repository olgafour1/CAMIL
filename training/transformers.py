from tensorflow.keras import layers
import tensorflow as tf
def mlp_block_f(mlp_dim, inputs):
    x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = layers.Dropout(rate=0.1)(x)  # dropout rate is from original paper,
    x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)  # check GELU paper
    x = layers.Dropout(rate=0.1)(x)
    return x


def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
    # self attention multi-head, dropout_rate is from original implementation
    x = layers.Add()([x, inputs])  # 1st residual part

    y = layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, y)
    y_1 = layers.Add()([y, x])  # 2nd residual part
    return y_1


def Encoder_f(num_layers, mlp_dim, num_heads, inputs):

    x = layers.Dropout(rate=0.2)(inputs)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = layers.LayerNormalization(name='encoder_norm')(x)
    return encoded