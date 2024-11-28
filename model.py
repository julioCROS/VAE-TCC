import gc
import torch
import auraloss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class VAE_GAN(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_dims, id, 
                 duration, rate, kernel_sizes, strides, loud_stride,
                 batch_size, residual_depth, num_bands=16, kl_beta=0.1, use_noise=False):
        print("[Incializando VAE-GAN]")
        super(VAE_GAN, self).__init__()
        # Definindo variaveis iniciais do modelo
        self.id = id
        self.rate = rate
        self.kl_beta = kl_beta
        self.strides = strides
        self.num_bands = num_bands
        self.duration = duration
        self.use_noise = use_noise
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.loud_stride = loud_stride
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.residual_depth = residual_depth
        
        # Construindo Encoder e Decoder do modelo
        self.build_encoder()
        self.build_decoder()
        print("[VAE-GAN Inicializado]")

    def build_encoder(self):
        # Construindo Encoder
        print("\t[CONSTRUINDO ENCODER]")
        inputs = layers.Input(shape=(self.input_shape[1], self.input_shape[2])) 
        x = inputs
        for h_dim, kernel, stride in zip(self.hidden_dims, self.kernel_sizes, self.strides):
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Conv1D(h_dim, kernel_size=kernel, strides=stride, padding='same')(x)

        x = layers.LeakyReLU(0.2)(x)
        mu = layers.Conv1D(filters=self.latent_dim, kernel_size=5, padding='same')(x)
        log_var = layers.Conv1D(filters=self.latent_dim, kernel_size=1, padding='same')(x)
        self.encoder = tf.keras.Model(inputs=inputs, outputs=[mu, log_var], name="Encoder")
        print("\t\t - Forma de saída MU: ", mu.shape)
        print("\t\t - Forma de saída LOG_VAR: ", log_var.shape)

    def build_decoder(self):
        print("\n\t[CONSTRUINDO DECODER]")

        factor = np.prod(self.strides)  # Fator acumulado das strides
        self.encoder_length = self.encoder.output_shape[0][1]
        total_decoder_units = self.hidden_dims[-1] * self.encoder_length  # Total de unidades geradas

        print("\t[INFO]Formato de entrada: ", self.input_shape)
        print("\t[INFO]Comprimento do Encoder: ", self.encoder_length)
        print("\t[INFO]Fator de Decodificação: ", factor)
        print("\t[INFO]Unidades do Decoder: ", total_decoder_units)

        # Entrada do decoder
        inputs = layers.Input(shape=(self.encoder_length, self.latent_dim))
        x = inputs

        # Decodificação
        for h_dim, kernel, stride in zip(self.hidden_dims[::-1], self.kernel_sizes, self.strides):
            x = UpsamplingBlock(h_dim, kernel, stride)(x)
            x = ResidualBlock(h_dim, kernel, self.residual_depth)(x)

        # Waveform e Loudness
        waveform_layer = layers.Conv1D(self.num_bands, kernel_size=7, padding='same', activation='tanh')
        waveform = waveform_layer(x)
        loudness_layer = layers.Conv1D(self.num_bands, kernel_size=2 * self.loud_stride + 1, 
                                    strides=self.loud_stride, padding='same', activation='sigmoid')
        loudness = loudness_layer(x)
        output = layers.Multiply()([waveform, loudness])

        # Adicionando ruído, se configurado
        if self.use_noise:
            noisesynth = NoiseSynthBlock(hidden_dims=self.hidden_dims, kernel_size=3)(x)
            output = layers.Add()([output, noisesynth])

        # Ajustar o comprimento da saída final
        output_length = self.input_shape[1]
        current_length = output.shape[1]

        if current_length < output_length:
            print(f"\t[INFO] Padding necessário ({current_length} != {output_length})")
            padding_needed = output_length - current_length
            output = layers.ZeroPadding1D(padding=(0, padding_needed))(output)
        elif current_length > output_length:
            print(f"\t[INFO] Cropping necessário ({current_length} != {output_length})")
            cropping_needed = current_length - output_length
            output = layers.Cropping1D(cropping=(0, cropping_needed))(output)

        self.decoder = tf.keras.Model(inputs, output, name="Decoder")
        print("\t\t - Forma de saída DECODER: ", output.shape)

    def call(self, inputs):
        # Chamada padrão do modelo
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        outputs = self.decode(z)
        return outputs, inputs, mu, log_var

    def encode(self, input):
        mu, log_var = self.encoder(input)
        log_var = tf.nn.softplus(log_var)
        return mu, log_var

    def reparameterize(self, mu, log_var):
      eps = tf.random.normal(shape=tf.shape(mu))
      z = mu + tf.exp(log_var * 0.5) * eps
      return z

    def decode(self, z):
      x = self.decoder(z)
      return x

    def train(self, data, epochs, optimizer):
      # Treinamento de representação do modelo
      print("[Iniciando Treinamento de Representação]")
      
      if self.batch_size == -1:
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(data.shape[0])
        print("[INFO] Dataset não dividido em Batches para o treinamento.\n")
      else: 
        print("[INFO] Dividindo dataset em batches para o treinamento.")
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        print(f"[INFO] Dataset dividido em batches de tamanho {self.batch_size} para o treinamento.\n")

      reconstruction_losses = []
      kl_losses = []

      for epoch in range(epochs):
        curr_batch = 0
        total_loss = 0; total_rec_loss = 0; total_kl_loss = 0
        for batch_data in dataset:
          loss, reconstruction_loss, kl_loss = self._train_step(batch_data, optimizer) 
          total_loss = total_loss + loss.numpy()
          total_rec_loss = total_rec_loss + reconstruction_loss.numpy()
          total_kl_loss = total_kl_loss + kl_loss.numpy()
        '''
          if self.batch_size != -1 and epoch % 15 == 0:
            curr_batch += 1
            print(f"\t[ Batch {curr_batch} / {int(data.shape[0]/self.batch_size)} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}]")
        '''
        if epoch % 15 == 0:
            print(f"# [ Epoca {epoch+1} | Loss: {np.around(total_loss, 7)} |  Recon. Loss: {np.around(total_rec_loss, 7)} | KL Loss: {np.around(total_kl_loss, 7)}]") 
        reconstruction_losses.append(total_rec_loss)
        kl_losses.append(total_kl_loss)
        gc.collect()
      return reconstruction_losses, kl_losses

    def _train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            outputs, inputs, mu, log_var = self(data)
            loss, reconstruction_loss, kl_loss = self._loss_function(inputs, outputs, mu, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, reconstruction_loss, kl_loss

    def _loss_function(self, inputs, outputs, mu, log_var):
        # Transpor para (batch_size, time_steps)
        inputs = tf.transpose(inputs, perm=[0, 2, 1])  
        outputs = tf.transpose(outputs, perm=[0, 2, 1])

        spectral_loss = self._multiscale_spectral_loss(inputs, outputs)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        total_loss = spectral_loss + self.kl_beta * kl_loss
        return total_loss, spectral_loss, kl_loss

    def _multiscale_spectral_loss(self, inputs, outputs, scales=[2048, 1024, 512, 256, 128]):
        def _compute_stft_loss(x, y, n_fft):
            # Calcula a STFT
            stft_x = tf.signal.stft(x, frame_length=n_fft, frame_step=n_fft // 4, fft_length=n_fft)
            stft_y = tf.signal.stft(y, frame_length=n_fft, frame_step=n_fft // 4, fft_length=n_fft)

            # Magnitude do espectro
            stft_x_mag = tf.cast(tf.abs(stft_x), dtype=tf.float32)
            stft_y_mag = tf.cast(tf.abs(stft_y), dtype=tf.float32)

            # Frobenius Loss
            frobenius_loss = tf.sqrt(tf.reduce_sum(tf.square(stft_x_mag - stft_y_mag))) / tf.sqrt(tf.reduce_sum(tf.square(stft_x_mag)))

            # L1 Loss
            l1_loss = tf.math.log(tf.reduce_mean(tf.abs(stft_x_mag - stft_y_mag)))

            # Retorna a soma das perdas
            return frobenius_loss + l1_loss

        # Combina perdas em múltiplas escalas
        total_loss = 0
        for scale in scales:
            total_loss += _compute_stft_loss(inputs, outputs, n_fft=scale)

        return total_loss / len(scales)

    def train_gan(self, data, epochs, gen_optimizer, discr_optimizer, hidden_dims, kernel_sizes, strides):
        print("[Iniciando Ajuste Fino Adversarial]")
        
        # Inicializar o Discriminador
        discriminator = Discriminator(hidden_dims, kernel_sizes, strides)

        # Criar dataset
        if self.batch_size == -1:
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(data.shape[0])
            print("[INFO] Dataset não dividido em Batches para o treinamento.\n")
        else: 
            print("[INFO] Dividindo dataset em batches para o treinamento.")
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
            print(f"[INFO] Dataset dividido em batches de tamanho {self.batch_size} para o treinamento.\n")

        # Armazenar perdas
        generator_losses = []
        discriminator_losses = []

        for epoch in range(epochs):
            curr_batch = 0
            generator_loss_epoch = 0
            discriminator_loss_epoch = 0

            for real_data in dataset:
                with tf.GradientTape(persistent=True) as tape_gen, tf.GradientTape(persistent=True) as tape_disc:
                    # Forward pass
                    fake_data, _, _, _ = self.call(real_data)

                    real_logits, real_features = discriminator(real_data, return_features=True)
                    fake_logits, fake_features = discriminator(fake_data, return_features=True)
                    disc_loss = self._discriminator_loss(real_logits, fake_logits)  # Perda do discriminador

                    # Transpor dados
                    real_data = tf.transpose(real_data, perm=[0, 2, 1])  
                    fake_data = tf.transpose(fake_data, perm=[0, 2, 1])

                    # Perdas
                    gen_loss_adv = self._generator_loss(fake_logits)  # Perda adversarial do gerador
                    spectral_loss = self._multiscale_spectral_loss(real_data, fake_data)  # Perda espectral (S(x, Ŷ))
                    fm_loss = self._feature_matching_loss(real_features, fake_features)  # Feature Matching
                    gen_loss = gen_loss_adv + fm_loss + spectral_loss  # Perda total do gerador

                # Backpropagation
                gen_grads = tape_gen.gradient(gen_loss, self.decoder.trainable_variables)
                disc_grads = tape_disc.gradient(disc_loss, discriminator.trainable_variables)
                gen_optimizer.apply_gradients(zip(gen_grads, self.decoder.trainable_variables))
                discr_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

                # Armazenar perdas
                generator_loss_epoch += gen_loss.numpy()
                discriminator_loss_epoch += disc_loss.numpy()
                
                '''
                if self.batch_size != -1 and epoch % 15 == 0:
                    curr_batch += 1
                    print(f"\t[ Batch {curr_batch} / {int(data.shape[0]/self.batch_size)} | "
                        f"Generator Loss: {gen_loss.numpy()} | Discriminator Loss: {disc_loss.numpy()} ]")
                '''

            # Perda por época
            if epoch % 15 == 0:
                print(f"# [Epoca {epoch + 1} | Generator Loss: {generator_loss_epoch} | Discriminator Loss: {discriminator_loss_epoch}]")
            generator_losses.append(generator_loss_epoch)
            discriminator_losses.append(discriminator_loss_epoch)
            gc.collect()
        return generator_losses, discriminator_losses


    def _generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)  # -E[D(Ŷ)]

    def _discriminator_loss(self, real_logits, fake_logits):
        real_loss = tf.nn.relu(1.0 - real_logits) # max(0, 1 - D(x))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))  # E [max(0, 1 + D(Ŷ))]
        return tf.reduce_mean(real_loss + fake_loss)
    
    def _feature_matching_loss(self, real_features, fake_features):
        fm_loss = 0
        for real_f, fake_f in zip(real_features, fake_features):
            fm_loss += tf.reduce_mean(tf.abs(real_f - fake_f))  # Feature Matching Loss
        return fm_loss / len(real_features)
    
    def compact_latent_representation(self, data, fidelity_threshold=0.95):
        # Calcular a média e centralizar os dados
        mu = np.mean(data, axis=0)  # Média sobre o batch
        mu_centered = data - mu    # Centralizar os dados
        
        # Remodelar para SVD
        reshaped_mu = mu_centered.reshape(-1, mu_centered.shape[-1])  # Forma: (batch_size * latent_dim, feature_dim)
        
        # Calcular SVD
        U, S, Vt = np.linalg.svd(reshaped_mu, full_matrices=False)

        # Calcular variância acumulada e número de dimensões informativas
        variance_explained = np.cumsum(S**2) / np.sum(S**2)
        num_informative_dims = np.searchsorted(variance_explained, fidelity_threshold) + 1
        self.num_informative_dims = num_informative_dims
        print(f"[Dimensões informativas selecionadas: {num_informative_dims} de {self.latent_dim}]\n")

        # Selecionar dimensões informativas
        self.informative_dimensions = Vt[:num_informative_dims, :]  # (num_informative_dims, feature_dim)
        print(f"[Dimensões informativas: {self.informative_dimensions.shape}]")
        
        # Reduzir a representação latente
        reduced_latent = np.matmul(reshaped_mu, Vt.T[:, :num_informative_dims])  # (batch_size * latent_dim, num_informative_dims)
        reduced_latent = reduced_latent.reshape(data.shape[0], data.shape[1], -1)  # Reshape para formato original reduzido
        
        return reduced_latent

    def _encode_compact(self, data, informative_dimensions):
        # Obtém a média do espaço latente
        mu, _ = self.encode(data)  # mu: (batch_size, 1, latent_dim)

        # Centraliza os dados
        mu_centered = mu - tf.reduce_mean(mu, axis=0)  # mu_centered: (batch_size, 1, latent_dim)

        # Validação de compatibilidade dimensional
        if mu_centered.shape[-1] != informative_dimensions.shape[-1]:
            raise ValueError(
                f"As dimensões não são compatíveis para a projeção: "
                f"mu_centered.shape={mu_centered.shape}, "
                f"informative_dimensions.shape={informative_dimensions.shape}"
            )

        # Projeta nas dimensões informativas
        compact_latent = tf.linalg.matmul(mu_centered, informative_dimensions, transpose_b=True)
        return compact_latent
    
    def sample(self, num_samples, data, compact_latent_space):
        if compact_latent_space is False:
            print("[Amostragem de espaço latente COMPLETO]")
            z = tf.random.normal(shape=(num_samples, self.encoder_length, self.latent_dim))
            return self.decode(z)
        else:
            # Verificar se o espaço latente compacto está configurado
            print("[Amostragem de espaço latente COMPACTO]")
            if not hasattr(self, 'informative_dimensions') or not hasattr(self, 'num_informative_dims'):
                raise ValueError("\t[ERROR]O espaço latente compacto não está configurado. Certifique-se de executar compact_latent_representation antes.")
            
            # Gerar espaço latente compacto
            compact_latent = self._encode_compact(data, self.informative_dimensions)

            print(f"\t[INFO]Formato do espaço latente compacto: {compact_latent.shape}")
            print(f"\t[INFO]Dimensões informativas: {self.num_informative_dims}")

            # Seleciona amostras aleatórias do espaço latente compacto
            random_indices = tf.random.uniform(shape=(num_samples,), minval=0, maxval=tf.shape(compact_latent)[0], dtype=tf.int32)
            sampled_latent = tf.gather(compact_latent, random_indices, axis=0)

            print(f"\t[INFO]Amostras de espaço latente compacto: {sampled_latent.shape}")

            # Ajusta para a dimensão completa do espaço latente original, preenchendo com zeros
            sampled_latent_padded = tf.pad(
                sampled_latent,
                paddings=[[0, 0], [0, 0], [0, self.latent_dim - self.num_informative_dims]],
            )
            sampled_latent_padded = tf.reshape(sampled_latent_padded, (num_samples, self.encoder_length, self.latent_dim))

            # Decodifica as amostras ajustadas
            generated_samples = self.decode(sampled_latent_padded)
            print(f"\t[INFO]Amostras finais geradas: {generated_samples.shape}")
            return generated_samples

    def generate(self, x):
        return self.call(x)[0]

# Implementação do bloco WN do Torch
class WeightNormalization(layers.Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape) 
        kernel = self.layer.kernel
        bias = self.layer.bias.numpy()
        self.v = kernel / tf.sqrt(tf.reduce_sum(tf.square(kernel), axis=[0, 1], keepdims=True))
        self.g = tf.Variable(tf.sqrt(tf.reduce_sum(tf.square(kernel))), trainable=True, name="g")
        normalized_weights = self.v * self.g
        self.layer.set_weights([normalized_weights, bias])  

    def call(self, inputs):
        return self.layer(inputs)

# Blocos auxiliares para construção do Variational AutoEncoder
class UpsamplingBlock(layers.Layer):
    def __init__(self, h_dim, kernel, stride, **kwargs):
        super(UpsamplingBlock, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.kernel = kernel
        self.stride = stride

    def call(self, x):
      x = layers.LeakyReLU(0.2)(x)
      x = layers.Conv1DTranspose(self.h_dim, kernel_size=self.kernel, strides=self.stride, padding='same')(x)
      return x
  
class ResidualBlock(layers.Layer):
    def __init__(self, h_dim, kernel, depth, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.kernel = kernel
        self.depth = depth

        self.conv_layers = [
            layers.Conv1D(h_dim, kernel_size=kernel, padding='same')
            for _ in range(depth)
        ]

        self.activation = layers.LeakyReLU(0.2)

    def call(self, x):
        res = x
        for conv in self.conv_layers:
            y = self.activation(res)
            y = conv(y) 
            res = layers.add([res, y])
        return res

class NoiseSynthBlock(layers.Layer):
  def __init__(self, hidden_dims, kernel_size=3, **kwargs):
      super(NoiseSynthBlock, self).__init__(**kwargs)
      self.hidden_dims = hidden_dims
      self.kernel_size = kernel_size
      self.num_filters = hidden_dims[-1]
      self.filter_noise = FilterNoiseBlock(num_filters=self.num_filters, kernel_size=kernel_size)
         
  def call(self, x):
      for h_dim in self.hidden_dims:
            x = layers.Conv1D(h_dim, kernel_size=self.kernel_size, padding='same')(x)
            x = layers.LeakyReLU()(x)

      white_noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
      filtered_noise = self.filter_noise(x, white_noise)
      return filtered_noise

class FilterNoiseBlock(layers.Layer):
    def __init__(self, num_filters, kernel_size=3, **kwargs):
        super(FilterNoiseBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def call(self, x, white_noise):
        x = layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.LeakyReLU()(x)
        filtered_noise = x * white_noise
        return filtered_noise

class Discriminator(tf.keras.Model):
    def __init__(self, hidden_dims, kernel_sizes, strides):
        super(Discriminator, self).__init__()
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Definindo as camadas convolucionais
        self.conv_layers = [
            layers.Conv1D(h_dim, kernel_size=k, strides=s, padding="same")
            for h_dim, k, s in zip(self.hidden_dims, self.kernel_sizes, self.strides)
        ]
        self.activation = layers.LeakyReLU()  # Ativação LeakyReLU
        self.global_avg_pooling = layers.GlobalAveragePooling1D()  # Pooling global
        self.output_layer = layers.Dense(1)  # Camada densa para logits finais

    def call(self, inputs, return_features=False):
        """
        Argumentos:
        - inputs: Tensor de entrada.
        - return_features: Booleano indicando se deve retornar as features intermediárias.

        Retorna:
        - Se `return_features` for True: (logits, features)
        - Caso contrário: logits
        """
        x = inputs
        features = []  # Lista para armazenar as features intermediárias

        # Passa os dados pelas camadas convolucionais
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
            features.append(x)  # Armazena a ativação intermediária

        # Global Average Pooling
        x = self.global_avg_pooling(x)

        # Camada de saída
        logits = self.output_layer(x)

        # Retorna features intermediárias, se solicitado
        if return_features:
            return logits, features
        return logits

