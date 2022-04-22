import tensorflow as tf
from preprocess import MAX_SMILE_LENGTH


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()  # TODO: IMPLEMENT ALL THE SIZES FOUND HERE
        self.input_size = 0
        self.latent_size = 0
        self.hidden_dim = 128
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(activation='selu'),
            tf.keras.layers.Conv1D(activation='selu'),
            tf.keras.layers.Conv1D(activation='selu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(activation='selu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(activation='selu'),
            tf.keras.layers.GRU(),
            tf.keras.layers.Dense(activation='softmax')
        ])
        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

    def call(self, input):
        encoder_out = self.encoder(input)
        mu = self.mu_layer(encoder_out)
        logvar = self.logvar_layer(encoder_out)
        latent_rep = self.reparametrize(mu, logvar)
        decoder_out = self.decoder(latent_rep)
        return decoder_out, mu, logvar

    def loss(self, decoder_out, input, mu, logvar):  # implementation of VAE loss; combo of reconstruction and KL
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM,)
        reconstruction_loss = bce_loss(input, decoder_out) * input.shape[-1]
        kl_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mu) - tf.math.exp(logvar))
        return (reconstruction_loss + kl_loss) / input.shape[0]

    def reparametrize(self, mu, logvar):
        epsilon = tf.random.normal((mu.shape[0], mu.shape[1]))
        sigma = tf.math.sqrt(tf.math.exp(logvar))
        return sigma * epsilon + mu
