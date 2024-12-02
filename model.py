import gc
import numpy as np
import loudness as ld
import tensorflow as tf
import audio_info as ainfo
import discriminator as discriminator
from tensorflow.keras import layers

class VAE_GAN(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, hidden_dims, id, 
                 duration, rate, kernel_sizes, strides, loud_stride,
                 discr_hidden_dims, discr_kernel_sizes, discr_strides,
                 batch_size, residual_depth, epoch_print = 1, delta = 0.1,
                 patience = 15, min_kl_loss = 0.1, num_bands=16, kl_beta=0.1,
                 use_noise=False, ld_block_size=512, mode="hinge", warmup=1000,
                 warmed_up=False):
        print("[Incializando VAE-GAN]")
        super(VAE_GAN, self).__init__()
        # Definindo variaveis iniciais do modelo
        self.id = id
        self.rate = rate
        self.mode = mode
        self.kl_beta = kl_beta
        self.strides = strides
        self.delta = delta
        self.warmpup = warmup
        self.warmed_up = warmed_up
        self.patience = patience
        self.min_kl_loss = min_kl_loss
        self.num_bands = num_bands
        self.duration = duration
        self.use_noise = use_noise
        self.ld_block_size = ld_block_size
        self.loudness = ld(self.rate, self.ld_block_size)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.loud_stride = loud_stride
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.epoch_print = epoch_print
        self.kernel_sizes = kernel_sizes
        self.residual_depth = residual_depth
        
        # Construindo Encoder e Decoder do modelo
        self.build_encoder()
        self.build_decoder()

        # Construindo Discriminador
        self.discriminator = discriminator(hidden_dims=discr_hidden_dims, kernel_sizes=discr_kernel_sizes, strides=discr_strides)
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
    
    def train(self, data, epochs, gen_optimizer, discr_optimizer):
        # Treinamento do modelo
        print("[Iniciando Treinamento de Representação]")
        if self.batch_size == -1:
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(data.shape[0])
            print("[INFO] Dataset não dividido em Batches para o treinamento.\n")
        else: 
            print("[INFO] Dividindo dataset em batches para o treinamento.")
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
            print(f"[INFO] Dataset dividido em {len(dataset)} batches de tamanho {self.batch_size} para o treinamento.\n")

        total_batches = len(dataset)
        for epoch in range(epochs):
            for batch_idx, data in enumerate(dataset):
                self._training_step(total_batches, batch_idx, data, epoch, gen_optimizer, discr_optimizer)
        print("[Treinamento concluído]")
        print("\t[Salvando modelo]")
        self.save(f"./models/{self.id}_epoch_{epoch}_batch_{batch_idx}.h5")
        print("\t[Modelo salvo]")                   

    def _training_step(self, total_batches, batch_idx, data, epoch, gen_optimizer, discr_optimizer, warmup = False):
        step = total_batches * epoch + batch_idx
        self.warmed_up = warmed_up = step > self.warmup
        if warmed_up:
            print("\n[VAE-GAN Aquecido]")
            print("[Iniciando treinamento GAN]\n")

        step_losses = {}

        # 1. Treinamento do VAE (Encoder + Decoder)
        inputs = data
        spectral_loss, kl_loss, outputs = self._training_step_vae(inputs=inputs, optimizer=gen_optimizer, warmup=warmup)

        distance = spectral_loss

        reconstructed_inputs = []
        reconstructed_outputs = []
        
        for input_multiband, output_multiband in zip(inputs, outputs):
            reconstructed_input = ainfo.multiband_synthesis(input_multiband, num_bands=self.num_bands)
            reconstructed_output = ainfo.multiband_synthesis(output_multiband, num_bands=self.num_bands)
            reconstructed_inputs.append(reconstructed_input)
            reconstructed_outputs.append(reconstructed_output)
        reconstructed_inputs = np.array(reconstructed_inputs)
        reconstructed_outputs = np.array(reconstructed_outputs)

        # Calcular a distância entre os sinais reconstruídos
        distance = distance + self._multiscale_spectral_loss(inputs=reconstructed_inputs, outputs=reconstructed_outputs)

        # Calcular a distância Loudness entre os sinais reconstruídos
        loud_x = self.loudness(reconstructed_inputs)
        loud_y = self.loudness(reconstructed_outputs)
        loud_dist = tf.reduce_mean(tf.math.pow(loud_x - loud_y, 2))
        distance = distance + loud_dist

        # 2. Treinamento do GAN (Gerador/Decoder + Discriminador)
        # Se não estiver aquecendo
        if warmed_up is True:
            distance, loss_adv, loss_dis, pred_true, pred_fake = self._training_step_gan(inputs=data, outputs=outputs, distance=distance)
        else:
            loss_adv = 0; loss_dis = 0; pred_true = 0; pred_fake = 0

        # Perda total do Decoder(Gerador)
        loss_gen = distance + loss_adv + self.kl_beta * kl_loss

        step_losses['spectral_loss'].append(spectral_loss)
        step_losses['kl_loss'].append(kl_loss)
        step_losses['loss_gen'].append(loss_gen)
        step_losses['loss_dis'].append(loss_dis)
        step_losses['pred_true'].append(pred_true.mean())
        step_losses['pred_Fake'].append(pred_fake.mean())

        # Aplicar otimização/gradientes dependendo do step
        if warmed_up is False:
            with tf.GradientTape() as vae_tape:
                vae_tape.watch(self.encoder.trainable_variables + self.decoder.trainable_variables)
                gradients = vae_tape.gradient(loss_gen, self.encoder.trainable_variables + self.decoder.trainable_variables)
                gen_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
        elif (step % 2 == 0) and warmed_up:
            with tf.GradientTape() as discr_tape:
                discr_tape.watch(self.discriminator.trainable_variables)
                gradients = discr_tape.gradient(loss_dis, self.discriminator.trainable_variables)
                discr_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        else:
            with tf.GradientTape() as gen_tape:
                gen_tape.watch(self.decoder.trainable_variables)
                gradients = gen_tape.gradient(loss_gen, self.decoder.trainable_variables)
                gen_optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))     

        # Printar perdas
        print(f"[Epoch: {epoch}] : [Step {step}]")
        print(f"\t Loss Gen.: {loss_gen}")
        print(f"\t Loss Dis.: {loss_dis}")
        print(f"\t Distance: {distance}")
        print(f"\t Loudness Dist.: {loud_dist}")
        print(f"\t Regularization Loss (KL): {kl_loss}")
        print(f"\t Pred. True: {pred_true.mean()}")
        print(f"\t Pred. Fake: {pred_fake.mean()}")
        gc.collect()
        return step_losses

    def _training_step_vae(self, inputs):
        # Treinamento de Representação do modelo
        outputs, inputs, mu, log_var = self(inputs)
        spectral_loss = self._multiscale_spectral_loss(inputs=inputs, outputs=outputs)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        return spectral_loss, kl_loss, outputs
    
    def _training_step_gan(self, inputs, outputs, distance):
        """
        Função para realizar um passo de treinamento adversarial.
        Calcula distance (Feature Matching Loss), loss_adv (Adversarial Loss),
        e loss_dis (Discriminator Loss).

        Argumentos:
        - inputs: Tensores reais (x).
        - outputs: Tensores gerados (y).

        Retorna:
        - distance: Perda de correspondência de features.
        - loss_adv: Perda adversarial.
        - loss_dis: Perda do discriminador.
        """
        # Sinais reais (x) e gerados (y)
        x = inputs
        y = outputs

        # Obter features e logits do discriminador
        logits_true, feature_true = self.discriminator(x, return_features=True)
        logits_fake, feature_fake = self.discriminator(y, return_features=True)

        # Inicializar métricas
        loss_adv = 0; loss_dis = 0; pred_true = 0; pred_fake = 0

        # Loop sobre as escalas do discriminador
        for scale_true, scale_fake in zip(feature_true, feature_fake):
            # Calcula a Feature Matching Loss
            distance += 10 * sum(
                tf.reduce_mean(tf.abs(x - y))
                for x, y in zip(scale_true, scale_fake)
            ) / len(scale_true)

            # Calcula adversarial_loss e discriminator_loss
            _dis, _adv = self._adversarial_combine(
                scale_true[-1], scale_fake[-1], mode=self.mode
            )

            pred_true += tf.reduce_mean(scale_true[-1])
            pred_fake += tf.reduce_mean(scale_fake[-1])
            loss_dis += _dis
            loss_adv += _adv

        # Retornar valores calculados
        return distance, loss_adv, loss_dis, pred_true, pred_fake
    
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
            l1_loss = tf.math.log(tf.reduce_sum(tf.abs(stft_x_mag - stft_y_mag)) + 1e-6)

            # Retorna a soma das perdas
            return frobenius_loss + l1_loss
        
        # Transpor para (batch_size, time_steps)
        inputs = tf.transpose(inputs, perm=[0, 2, 1])  
        outputs = tf.transpose(outputs, perm=[0, 2, 1])

        # Combina perdas em múltiplas escalas
        total_loss = 0

        for scale in scales:
            total_loss += _compute_stft_loss(inputs, outputs, n_fft=scale)

        return total_loss / len(scales)
    
    def _adversarial_combine(self, score_real, score_fake, mode="hinge"):
        """
        Combina as pontuações reais e falsas do discriminador para calcular as perdas adversariais.

        Argumentos:
        - score_real: Tensor de pontuações reais do discriminador.
        - score_fake: Tensor de pontuações falsas do discriminador.
        - mode: String indicando o tipo de perda adversarial ("hinge" ou "square").

        Retorna:
        - loss_dis: Perda do discriminador.
        - loss_gen: Perda do gerador.
        """
        if mode == "hinge":
            # Perda do discriminador
            loss_dis = tf.nn.relu(1 - score_real) + tf.nn.relu(1 + score_fake)
            loss_dis = tf.reduce_mean(loss_dis)

            # Perda do gerador
            loss_gen = -tf.reduce_mean(score_fake)

        elif mode == "square":
            # Perda do discriminador
            loss_dis = tf.square(score_real - 1) + tf.square(score_fake)
            loss_dis = tf.reduce_mean(loss_dis)

            # Perda do gerador
            loss_gen = tf.reduce_mean(tf.square(score_fake - 1))

        else:
            raise NotImplementedError(f"Modo adversarial '{mode}' não implementado.")

        return loss_dis, loss_gen
    
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

