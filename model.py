import tensorflow as tf
from preprocess import MAX_SMILE_LENGTH


class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()  # TODO: implement these sizes
        self.c1 = tf.keras.layers.Conv1D(activation='selu')
        self.c2 = tf.keras.layers.Conv1D(activation='selu')
        self.c3 = tf.keras.layers.Conv1D(activation='selu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(activation='selu')
        self.mu = tf.keras.layers.Dense()  # TODO: implement random sampling of mu and logvar like VAE
        self.logvar = tf.keras.layers.Dense()

    def call(self, input):
        out = self.c1(input)
        out = self.c2(out)
        out = self.c3(out)
        out = self.flatten(out)
        out = self.dense(out)
        # TODO: get mu and logvar from lambda class like VAE assignment and return it

    def loss(self, x_hat, x, mu, logvar):  # implementation of VAE loss; combo of reconstruction and KL
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM,)
        reconstruction_loss = bce_loss(x, x_hat) * x.shape[-1]
        kl_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mu) - tf.math.exp(logvar))
        return (reconstruction_loss + kl_loss) / x.shape[0]


class Decoder(tf.keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()  # TODO: implement these sizes
        self.dense1 = tf.keras.layers.Dense(activation='selu')
        # self.repeat = Repeat()  # TODO: implement repeat
        self.gru = tf.keras.layers.GRU()
        self.dense2 = tf.keras.layers.Dense(activation='softmax')  # TODO: timedistributed here?

    def call(self, encoded_input):
        out = self.dense1(encoded_input)
        # out = self.repeat(out)
        out, _ = self.gru(out)
        return self.dense2(out)
