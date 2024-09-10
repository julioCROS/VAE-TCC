import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_dims=None, id=None, duration=None, rate=None, kl_annealing_rate=0.01, max_kl_weight=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.id = id
        self.duration = duration
        self.rate = rate
        self.hidden_dims = hidden_dims

        self.kl_weight = 0
        self.kl_annealing_rate = kl_annealing_rate
        self.max_kl_weight = max_kl_weight 

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.InputLayer(input_shape=(self.input_shape[1], self.input_shape[2], self.input_shape[3])))
        for h_dim in self.hidden_dims:
            self.encoder.add(layers.Conv2D(h_dim, kernel_size=(3,3), strides=(2,2), padding='same'))
            self.encoder.add(layers.LayerNormalization())
            self.encoder.add(layers.LeakyReLU())

        self.encoder.add(layers.GlobalAveragePooling2D())
        self.encoder.add(layers.Dense(self.latent_dim * 2))
        
        self.fc_mu = layers.Dense(self.latent_dim)
        self.fc_var = layers.Dense(self.latent_dim)

    def build_decoder(self):
        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.InputLayer(input_shape=(self.latent_dim,)))

        factor = 2 ** len(self.hidden_dims)
        units = self.hidden_dims[-1] * (self.input_shape[1] // factor) * (self.input_shape[2] // factor)

        self.decoder.add(layers.Dense(units, activation='linear'))
        self.decoder.add(layers.Reshape((self.input_shape[1] // factor, self.input_shape[2] // factor, self.hidden_dims[-1])))

        for h_dim in self.hidden_dims[::-1]:
            self.decoder.add(layers.Conv2DTranspose(h_dim, kernel_size=(3,3), strides=(2,2), padding='same'))
            self.decoder.add(layers.LayerNormalization())
            self.decoder.add(layers.LeakyReLU())

        self.decoder.add(layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        if self.decoder.output_shape[1] < self.input_shape[1]:
            padding_needed = self.input_shape[1] - self.decoder.output_shape[1]
            self.decoder.add(layers.ZeroPadding2D(padding=((padding_needed, 0), (0, 0))))
        elif self.decoder.output_shape[1] > self.input_shape[1]:
            cropping_needed = self.decoder.output_shape[1] - self.input_shape[1]
            self.decoder.add(layers.Cropping2D(cropping=((cropping_needed, 0), (0, 0))))
        
        if self.decoder.output_shape[2] < self.input_shape[2]:
            padding_needed = self.input_shape[2] - self.decoder.output_shape[2]
            self.decoder.add(layers.ZeroPadding2D(padding=((0, 0), (padding_needed, 0))))
        elif self.decoder.output_shape[2] > self.input_shape[2]:
            cropping_needed = self.decoder.output_shape[2] - self.input_shape[2]
            self.decoder.add(layers.Cropping2D(cropping=((0, 0), (cropping_needed, 0))))

        self.decoder.add(layers.Reshape((self.input_shape[1], self.input_shape[2], 1)))

    def call(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        outputs = self.decode(z)
        return outputs, inputs, mu, log_var

    def encode(self, input):
        x = self.encoder(input)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        z = eps * tf.exp(logvar * 0.5) + mu
        quantized_z = tf.round(z)  
        return quantized_z   

    def decode(self, z):
        x = self.decoder(z)
        return x

    def train(self, data, epochs, optimizer):
        best_epoch_metadata = None
        min_reconstruction_loss = float('inf')
        reconstruction_losses = []
        kl_losses = []
        for epoch in range(epochs):
            loss, reconstruction_loss, kl_loss = self.train_step(data, optimizer) 

            if self.kl_weight < self.max_kl_weight:
                self.kl_weight += self.kl_annealing_rate
                self.kl_weight = min(self.kl_weight, self.max_kl_weight)

            if reconstruction_loss < min_reconstruction_loss:
                min_reconstruction_loss = reconstruction_loss
                best_epoch_metadata = epoch, loss, reconstruction_loss, kl_loss, self.kl_weight

            reconstruction_losses.append(reconstruction_loss.numpy())
            kl_losses.append(kl_loss.numpy())
            print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")
        return best_epoch_metadata, reconstruction_losses, kl_losses
    
    @tf.function
    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            outputs, inputs, mu, log_var = self(data)
            loss, reconstruction_loss, kl_loss = self.loss_function(inputs, outputs, mu, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, reconstruction_loss, kl_loss        
    
    def loss_function(self, inputs, outputs, mu, log_var):
        assert inputs.shape == outputs.shape, f"Shape mismatch: {inputs.shape} vs {outputs.shape}"
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(inputs, outputs))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))

        latent_penalty = tf.reduce_mean(tf.abs(mu))
        total_loss = reconstruction_loss + (self.kl_weight * kl_loss) + 0.1 * latent_penalty
        # total_loss = reconstruction_loss + (self.kl_weight * kl_loss)
        return total_loss, reconstruction_loss, kl_loss
    
    def sample(self, num_samples):
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def generate(self, x):
        return self.call(x)[0]
