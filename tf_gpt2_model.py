import tensorflow as tf
import math


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((tf.sqrt(2 / math.pi)
                                * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def create_attention_mask(x):
    x = tf.reduce_any(x != 0, axis=-1)
    x = x[:, tf.newaxis, tf.newaxis, :]
    x = tf.cast(x, tf.float32)
    return (1 - x) * -1e4


class MLP(tf.keras.layers.Layer):
    def __init__(self, n_state, n_features, **kwargs):
        super().__init__(**kwargs)
        nx = n_features
        self.c_fc = Conv1D(n_state, nx, name="c_fc")
        self.c_proj = Conv1D(nx, n_state, name="c_proj")
        self.act = gelu
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, nf, nx, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx

    def build(self, input_shape):
        self.weight = self.add_weight(
            'weight',
            shape=[self.nx, self.nf],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.bias = self.add_weight(
            'bias',
            shape=[1, self.nf],
            initializer=tf.zeros_initializer(),
        )

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias
        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_features, n_heads, **kwargs):
        super().__init__(**kwargs)

        assert n_features % n_heads == 0
        self.n_heads = n_heads

        self.c_attn = Conv1D(n_features * 3, n_features, name='c_attn')
        self.c_proj = Conv1D(n_features, n_features, name='c_proj')
        self.attn_dropout = tf.keras.layers.Dropout(0.1)
        self.resid_dropout = tf.keras.layers.Dropout(0.1)

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, mask, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(shape_list(k)[-1], tf.float32)  # scale attention_scores
        w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence]
        # where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        # Apply the attention mask
        w = w + mask

        w = tf.nn.softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # return outputs and attn weights
        return tf.matmul(w, v), w

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + \
            [self.n_heads, x_shape[-1] // self.n_heads]
        x = tf.reshape(x, new_x_shape)
        # (batch, head, seq_length, head_features)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, mask, training=False):
        x = self.c_attn(x)
        q, k, v = tf.split(x, 3, axis=2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        a, w = self._attn(q, k, v, mask, training=training)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        return a, w


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_features, n_heads, scale=True, **kwargs):
        super().__init__(**kwargs)

        self.ln_1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name='ln_1',
        )
        self.attn = Attention(n_features, n_heads, name='attn')
        self.ln_2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name='ln_2',
        )
        self.mlp = MLP(4 * n_features, n_features, name='mlp')

    def call(self, x, mask, training=False):
        a = self.ln_1(x)
        a, w = self.attn(a, mask, training=training)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        return x, w


class MimicGpt2(tf.keras.Model):
    def __init__(self, n_days, n_features, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_days = n_days
        self.n_features = n_features
        self.n_ctx = 1024
        self.n_layers = 12

        self.wpe = tf.keras.layers.Embedding(
            n_days,
            n_features,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name='wpe'
        )
        self.drop = tf.keras.layers.Dropout(0.1)
        self.decoders = [
            Decoder(n_features, n_heads, name=f'decoder_{i}') for i in range(self.n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name='ln_f')
        self.linear = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name='sigmoid'
        )

    # TODO: try the correct order or layer norms
    def call(self, x, training=False):
        # sanity check
        _, n_days, n_features = shape_list(x)
        assert n_days == self.n_days, "x's shape should be [batch_size, n_days, n_features]"
        assert n_features == self.n_features, "x's shape should be [batch_size, n_days, n_features]"

        # retrieve position ids (actually the whole wpe kernel)
        position_ids = tf.range(self.n_days, dtype=tf.int32)
        position_ids = position_ids[tf.newaxis, :]

        # mask days with all zero values
        attn_mask = create_attention_mask(x)

        # add positional encodings and optional dropout
        x = x + self.wpe(position_ids)
        x = self.drop(x, training=training)

        all_attn_weights = []
        for decoder in self.decoders:
            x, w = decoder(x, attn_mask, training=training)

            # save attention weights for this decoder block
            all_attn_weights.append(w)

        # go through final layer norm
        x = self.ln_f(x)

        # project linearly to get probabilities for each day
        x = self.linear(x)

        return x, all_attn_weights
