from tensorflow.keras.layers import Dense, Layer, Masking, LSTM, TimeDistributed, Dropout
from tensorflow.keras import Model

import tensorflow as tf


def create_attention_mask(x):
    x = tf.reduce_any(x != 0, axis=-1)
    x = x[:, :, tf.newaxis]
    x = tf.cast(x, tf.float32)
    return (1 - x) * -1e9


class Attention(Layer):
    def __init__(self, time_steps=14, **kwargs):
        super().__init__(**kwargs)
        self.dense = Dense(
            time_steps,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02)
        )
        self.drop = Dropout(0.1)

    def call(self, x, mask):
        w = tf.transpose(x, perm=(0, 2, 1))
        w = self.dense(w)
        w = self.drop(w)
        w = tf.transpose(w, perm=(0, 2, 1))
        w = w + mask
        w = tf.nn.softmax(w, axis=-2)
        x = x * w
        return x, w


class Mimic3Lstm(Model):
    def __init__(self, time_steps=14, mask=False, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(time_steps=time_steps)
        self.masking = Masking(mask_value=0)
        self.lstm = LSTM(256, return_sequences=True)
        self.distributed = TimeDistributed(Dense(1, activation='sigmoid'))

    def call(self, x):
        mask = create_attention_mask(x)
        x, w = self.attention(x, mask)
        x = self.masking(x)
        x = self.lstm(x)
        x = self.distributed(x)
        return x, w
