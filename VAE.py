import tensorflow as tf
import torch
from tensorflow.keras import layers
import numpy as np
import auraloss

class VAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_dims, id, 
                 duration, rate, kernel_sizes, strides,
                kl_annealing_rate=0.01, max_kl_weight=1.0,):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.id = id
        self.duration = duration
        self.rate = rate
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.kl_weight = 0
        self.kl_annealing_rate = kl_annealing_rate
        self.max_kl_weight = max_kl_weight 

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.InputLayer(input_shape=(self.input_shape[1], self.input_shape[2], self.input_shape[3])))
        for h_dim in self.hidden_dims:
            self.encoder.add(layers.Conv2D(h_dim, kernel_size=self.kernel_sizes[0], strides=self.strides[0], padding='same'))
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
            self.decoder.add(layers.Conv2DTranspose(h_dim, kernel_size=self.kernel_sizes[1], strides=self.strides[1], padding='same'))
            self.decoder.add(layers.LayerNormalization())
            self.decoder.add(layers.LeakyReLU())

        self.decoder.add(layers.Conv2DTranspose(1, kernel_size=self.kernel_sizes[2], strides=self.strides[2], padding='same', activation='relu'))

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
    
    def representation_learning_train(self, data, epochs, optimizer):
        print("\n[Iniciando aprendizado de representação...]")
        spectral_losses = []
        repr_kl_losses = []
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                outputs, inputs, mu, log_var = self(data)
                spectral_loss = self.compute_spectral_loss(inputs, outputs)
                kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
                loss = spectral_loss + self.kl_weight * kl_loss
                
                print(f"Segmento {segment_count} de {len(data)}.")
                segment_count += 1

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            if self.kl_weight < self.max_kl_weight:
                self.kl_weight += self.kl_annealing_rate
                self.kl_weight = min(self.kl_weight, self.max_kl_weight)

            spectral_losses.append(spectral_loss.numpy())
            repr_kl_losses.append(kl_loss.numpy())
            print(f"[Epoca {epoch+1} | Spectral Loss: {spectral_loss.numpy()} | KL Loss: {kl_loss.numpy()} | Total Loss: {loss.numpy()}]")
        return spectral_losses, repr_kl_losses

    def compute_spectral_loss(self, real, fake):
        # Converte os tensores de TensorFlow para NumPy
        real_np = real.numpy()
        fake_np = fake.numpy()

        # Converte os arrays NumPy para tensores PyTorch
        real_torch = torch.from_numpy(real_np).float()
        fake_torch = torch.from_numpy(fake_np).float()

        real_torch = real_torch.permute(0, 2, 1)
        fake_torch = fake_torch.permute(0, 2, 1)

        # Calcula a perda STFT usando auraloss
        loss_fn = auraloss.time.SNRLoss()
        spectral_loss = loss_fn(real_torch, fake_torch)

        # Converte o resultado da perda de volta para TensorFlow
        return tf.convert_to_tensor(spectral_loss.item())

    def train(self, data, epochs, optimizer):
        print("\n[Iniciando treinamento...]")
        reconstruction_losses = []
        kl_losses = []
        for epoch in range(epochs):
            print(f"\n[Epoca {epoch+1}]")
            loss, reconstruction_loss, kl_loss = self.train_step(data, optimizer) 
            if self.kl_weight < self.max_kl_weight:
                self.kl_weight += self.kl_annealing_rate
                self.kl_weight = min(self.kl_weight, self.max_kl_weight)

            reconstruction_losses.append(reconstruction_loss.numpy())
            kl_losses.append(kl_loss.numpy())
            print(f"[ Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_losses.numpy()} | KL Loss: {kl_losses.numpy()}]")
        return reconstruction_losses, kl_losses
    
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
        return total_loss, reconstruction_loss, kl_loss
    
    def adversarial_fine_tuning_train(self, data, epochs, generator_optimizer, discriminator_optimizer, discriminator):
        print("Iniciando ajuste fino adversarial...")
        discriminator = Discriminator()
        for epoch in range(epochs):
            for real_data in data:
                with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
                    fake_data, _, _, _ = self(real_data)

                    real_logits = discriminator(real_data)
                    fake_logits = discriminator(fake_data)

                    gen_loss = -tf.reduce_mean(fake_logits)
                    disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits)) + tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))

                gen_grads = tape_gen.gradient(gen_loss, self.decoder.trainable_variables)
                disc_grads = tape_disc.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gen_grads, self.decoder.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            print(f"Epoch {epoch + 1} | Generator Loss: {gen_loss.numpy()} | Discriminator Loss: {disc_loss.numpy()}")
    
    def compact_latent_representation(self, data, fidelity_threshold=0.95):
        print("\n[Compactando a representação latente...]")
        mu, _ = self.encode(data)
        mu = mu.numpy() 

        mu_centered = mu - np.mean(mu, axis=0)

        U, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)
        variance_explained = np.cumsum(S**2) / np.sum(S**2)
        num_informative_dims = np.searchsorted(variance_explained, fidelity_threshold) + 1

        print(f"\t[Dimensões informativas selecionadas: {num_informative_dims} de {self.latent_dim}]\n")
        
        reduced_latent = np.dot(mu_centered, Vt.T[:, :num_informative_dims])
        return reduced_latent, Vt[:num_informative_dims]

    def encode_compact(self, data, informative_dimensions):
        mu, _ = self.encode(data)
        mu_centered = mu - np.mean(mu, axis=0)
        
        compact_latent = np.dot(mu_centered, informative_dimensions.T)
        return compact_latent
    
    def sample(self, num_samples, data, informative_dimensions, compact_latent_space):
        if compact_latent_space is False:
            z = tf.random.normal(shape=(num_samples, self.latent_dim))
            return self.decode(z)
        else:
            compact_latent = self.encode_compact(data, informative_dimensions)
            
            num_informative_dims = informative_dimensions.shape[1]  
            print(f"compact_latent shape: {compact_latent.shape}")
            print(f"num_informative_dims: {num_informative_dims}")
            
            random_indices = tf.random.uniform(shape=(num_samples,), minval=0, maxval=len(compact_latent), dtype=tf.int32)
            sampled_latent = tf.gather(compact_latent, random_indices)
            
            sampled_latent = tf.reshape(sampled_latent, (num_samples, num_informative_dims))
            
            generated_samples = self.decode(sampled_latent)
            return generated_samples
    
    def decode(self, z):
        x = self.decoder(z)
        return x

    def generate(self, x):
        return self.call(x)[0]
    
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

