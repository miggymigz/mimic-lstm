from tensorflow.keras.layers import Dense, LSTM, Masking, TimeDistributed, Layer
from tensorflow.keras import Model

import tensorflow as tf


class Attention(Layer):
    def __init__(self, time_steps=14, **kwargs):
        super().__init__(**kwargs)
        self.dense = Dense(time_steps, activation='softmax')

    def call(self, x):
        w = tf.transpose(x, perm=(0, 2, 1))
        w = self.dense(w)
        w = tf.transpose(w, perm=(0, 2, 1))
        x = x * w
        return x, w


class Mimic3BaseLstm(Model):
    def __init__(self, time_steps=14, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(time_steps=time_steps)
        self.masking = Masking(mask_value=0)
        self.lstm = LSTM(256, return_sequences=True)
        self.distributed = TimeDistributed(Dense(1, activation='sigmoid'))

    def call(self, inputs):
        x, w = self.attention(inputs)
        x = self.masking(x)
        x = self.lstm(x)
        x = self.distributed(x)
        return x, w
