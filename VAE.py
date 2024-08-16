import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_dims=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.InputLayer(input_shape=(self.input_shape[0], 1)))
        for h_dim in self.hidden_dims:
            self.encoder.add(layers.Conv1D(h_dim, kernel_size=3, strides=2, padding='same'))
            self.encoder.add(layers.BatchNormalization())
            self.encoder.add(layers.LeakyReLU())

        self.encoder.add(layers.GlobalAveragePooling1D())
        self.encoder.add(layers.Dense(self.latent_dim * 2))
        
        self.fc_mu = layers.Dense(self.latent_dim)
        self.fc_var = layers.Dense(self.latent_dim)

    def build_decoder(self):
        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.InputLayer(input_shape=(self.latent_dim,)))

        # Adjusting first Dense layer to match latent_dim
        self.decoder.add(layers.Dense(self.hidden_dims[-1] * (self.input_shape[1] // (2 ** len(self.hidden_dims))), activation='relu'))

        # Reshape to match the Conv1DTranspose layers
        self.decoder.add(layers.Reshape(((self.input_shape[1] // (2 ** len(self.hidden_dims))), self.hidden_dims[-1])))

        for h_dim in self.hidden_dims[::-1]:
            self.decoder.add(layers.Conv1DTranspose(h_dim, kernel_size=3, strides=2, padding='same'))
            self.decoder.add(layers.BatchNormalization())
            self.decoder.add(layers.LeakyReLU())

        self.decoder.add(layers.Conv1DTranspose(1, kernel_size=3, strides=1, padding='same'))

        # Ensure final output shape matches input shape
        if self.decoder.output_shape[1] < self.input_shape[1]:
            padding_needed = self.input_shape[1] - self.decoder.output_shape[1]
            self.decoder.add(layers.ZeroPadding1D(padding=(0, padding_needed)))
        elif self.decoder.output_shape[1] > self.input_shape[1]:
            cropping_needed = self.decoder.output_shape[1] - self.input_shape[1]
            self.decoder.add(layers.Cropping1D(cropping=(0, cropping_needed)))

        self.decoder.add(layers.Reshape((self.input_shape[1], 1)))

    def encode(self, input):
        x = self.encoder(input)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return eps * tf.exp(logvar * .5) + mu

    def call(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        outputs = self.decode(z)
        return outputs, inputs, mu, log_var

    def loss_function(self, inputs, outputs, mu, log_var):
        outputs = tf.squeeze(outputs, axis=-1) 
        assert inputs.shape == outputs.shape, f"Shape mismatch: {inputs.shape} vs {outputs.shape}"
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(inputs, outputs))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def sample(self, num_samples):
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def generate(self, x):
        return self.call(x)[0]
