import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_features, time_steps=14, **kwargs):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.time_steps = time_steps

        self.dense = tf.keras.layers.Dense(
            time_steps,
            activation='softmax',
            input_shape=(n_features, time_steps),
        )

    def call(self, inputs):
        a_weights = tf.transpose(inputs, perm=(0, 2, 1))
        a_weights = self.dense(a_weights)
        a_weights = tf.transpose(a_weights, perm=(0, 2, 1))
        x = tf.multiply(inputs, a_weights)
        return x, a_weights


class Mimic3Lstm(tf.keras.Model):
    def __init__(self, n_features, time_steps=14, batch_size=1, **kwargs):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.time_steps = time_steps

        self.attention = Attention(
            n_features,
            time_steps=time_steps,
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
        x, a_weights = self.attention(inputs)
        x = self.masking(x)
        x = self.lstm(x)
        preds = self.distributed(x)
        return preds, a_weights

    def model(self):
        x = tf.keras.layers.Input(shape=(self.time_steps, self.n_features))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
