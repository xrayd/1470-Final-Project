import tensorflow as tf
from preprocess import MAX_SMILE_LENGTH


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()  # TODO: IMPLEMENT ALL THE SIZES FOUND HERE
        self.smile_length = 80
        self.vocab_length = 52
        self.input_size = self.smile_length * self.vocab_length
        self.latent_size = 64
        self.hidden_dim = 64

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hidden_dim, input_shape=(1, self.input_size), activation='swish'),
            tf.keras.layers.Dense(self.hidden_dim, activation='swish'),
            tf.keras.layers.Dense(self.hidden_dim, activation='swish')
        ])
        self.decoder = tf.keras.Sequential([
            # Perhaps 292 by 435
            tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.latent_size,), activation='swish'),
            # tf.keras.layers.GRU(self.hidden_dim),
            tf.keras.layers.Dense(self.hidden_dim, activation='swish'),
            tf.keras.layers.Dense(self.input_size, activation='softmax'),
            tf.keras.layers.Reshape((1, self.smile_length, self.vocab_length))
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
