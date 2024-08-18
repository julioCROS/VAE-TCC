import numpy as np
import tensorflow as tf

from VAE import VAE
from AudioReader import AudioReader

# Parâmetros
audio_duration = 30; audio_rate = 11000;        # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'             # Caminho do arquivo de áudio
audio_data = AudioReader(audio_path)            # Leitor de áudio
latent_dim = 2                                  # Dimensão do espaço latente
batch_size = 16                                 # Tamanho do lote
epochs = 2                                      # Número de épocas

# Dados de Treinamento
x_train, sample_rate = audio_data.read_audio(duration= audio_duration, sr=audio_rate)   # (330000,)
x_train = x_train.reshape(1, x_train.shape[0], 1)                                       # (1, 330000, 1)

def create_batches(data, batch_size):
    num_samples = data.shape[1]
    num_batches = num_samples // batch_size
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = data[:, start:end, :] 
        batches.append(batch)
    batches = np.array(batches)
    return batches

x_train = create_batches(x_train, batch_size)                                            # (20625, 1, 16, 1)

# Salvando os dados de treinamento em um arquivo txt
#np.savetxt('./files/input_AUDIO.txt', x_train.reshape(audio_duration, audio_rate), fmt = "%f")

# Inicialização do Modelo
vae = VAE(input_shape = x_train.shape, latent_dim = latent_dim)
vae.summary()
#vae.encoder.summary()
#vae.decoder.summary()

# Treinamento do Modelo
optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        outputs, inputs, mu, log_var = vae(x)
        loss, reconstruction_loss, kl_loss = vae.loss_function(inputs, outputs, mu, log_var)
        print("\t - LOSS FUNCTION [COMPLETE]")
    gradients = tape.gradient(loss, vae.trainable_variables)
    print("\t - GRADIENTS [COMPLETE]")
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    print("\t - OPTIMIZER [COMPLETE]")
    return loss, reconstruction_loss, kl_loss

# Loop de Treinamento
best_output = None
min_reconstruction_loss = float('inf')
for epoch in range(epochs):
    print(f"\n - ### Epoca {epoch+1} de {epochs}")
    for i in range(len(x_train)):
        print(f"\t - ## Batch {i+1} de {len(x_train)}")
        x_batch = x_train[i]
        print(f"\t - X_BATCH Shape: {x_batch.shape}")
        loss, reconstruction_loss, kl_loss = train_step(x_batch)
        print("\t - TRAIN STEP [COMPLETE] ")
    if reconstruction_loss < min_reconstruction_loss:
        min_reconstruction_loss = reconstruction_loss
    print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")




