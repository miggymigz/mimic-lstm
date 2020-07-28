import tensorflow as tf


def create_attention_mask(x):
    x = tf.reduce_any(x != 0, axis=-1)
    x = x[:, :, tf.newaxis]
    x = tf.cast(x, tf.float32)
    return (1 - x) * -1e9


class Attention(tf.keras.layers.Layer):
    def __init__(self, time_steps=14, mask=False, **kwargs):
        super().__init__(**kwargs)

        self.mask = mask

        self.dense = tf.keras.layers.Dense(
            time_steps,
            activation=None,
        )

    def call(self, x):
        w = tf.transpose(x, perm=(0, 2, 1))
        w = self.dense(w)
        w = tf.transpose(w, perm=(0, 2, 1))

        # apply mask to padding days
        if self.mask:
            mask = create_attention_mask(x)
            w = w + mask

        # softmax scores and apply attention weights
        w = tf.nn.softmax(w, axis=-2)
        x = tf.multiply(x, w)

        return x, w


class Mimic3Lstm(tf.keras.Model):
    def __init__(self, n_features, time_steps=14, mask=False, batch_size=1, **kwargs):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.time_steps = time_steps

        self.attention = Attention(
            time_steps=time_steps,
            mask=mask,
            batch_input_shape=(batch_size, time_steps, n_features)
        )
        self.masking = tf.keras.layers.Masking(
            mask_value=0,
        )
        self.lstm = tf.keras.layers.LSTM(
            256,
            return_sequences=True,
        )
        self.distributed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='sigmoid')
        )

    def call(self, inputs):
        x, w = self.attention(inputs)
        x = self.masking(x)
        x = self.lstm(x)
        x = self.distributed(x)
        return x, w

    def model(self):
        x = tf.keras.layers.Input(shape=(self.time_steps, self.n_features))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
