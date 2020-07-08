import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_features, time_steps=14, **kwargs):
        super(Attention, self).__init__(**kwargs)

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
    def __init__(self, n_features, time_steps=14, learning_rate=0.001, optimizer='rms'):
        super(Mimic3Lstm, self).__init__()

        self.n_features = n_features
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.optimizer_t = optimizer

        self.attention = Attention(
            n_features,
            time_steps=time_steps,
            input_shape=(time_steps, n_features),
        )
        self.masking = tf.keras.layers.Masking(
            mask_value=0,
            input_shape=(time_steps, n_features),
        )
        self.lstm = tf.keras.layers.LSTM(
            256,
            return_sequences=True,
            input_shape=(time_steps, n_features),
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

    def create_optimizer(self):
        optimizer_t = self.optimizer_t.lower()
        if optimizer_t == 'rms':
            return tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate,
                rho=0.9,
                epsilon=1e-08,
            )

        if optimizer_t == 'adam':
            return tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
            )

        raise AssertionError(
            f'ERROR: Optimizer "{self.optimizer}" is not supported')

    def model(self):
        x = tf.keras.layers.Input(shape=(self.time_steps, self.n_features))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
