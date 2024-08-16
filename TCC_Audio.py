import numpy as np
import tensorflow as tf

from VAE import VAE
from AudioReader import AudioReader

# Parâmetros
audio_duration = 1; audio_rate = 1000;
audio_path = './audio/ethereal.ogg'
audio_data = AudioReader(audio_path)

print("\n#################################################")
print("  Iniciando o Processo de Treinamento... \n")
print(" - Duração do Áudio:", audio_duration, "segundos")
print(" - Taxa de Amostragem:", audio_rate, "Hz")
print("#################################################\n")


latent_dim = 64                                  # Dimensão do espaço latente
batch_size = 16                                  # Tamanho do lote
epochs = 2                                       # Número de épocas

# Dados de Treinamento
#x_train, sample_rate = audio_data.read_audio(duration= audio_duration, sr=audio_rate)
#x_train = x_train.reshape(x_train.shape[0], 1)
x_train = np.random.rand(1000, 1) 

# Salvando os dados de treinamento em um arquivo txt
np.savetxt('./files/input_AUDIO.txt', x_train.reshape(audio_duration, audio_rate), fmt = "%f")

# Inicialização do Modelo
vae = VAE(input_shape = x_train.shape, latent_dim = latent_dim)

# Compilação do Modelo
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
    print("\t\t - LOSS: ", loss)
    print("\t\t - RECONSTRUCTION LOSS: ", reconstruction_loss)
    print("\t\t - KL LOSS: ", kl_loss)
    return loss, reconstruction_loss, kl_loss

# Loop de Treinamento
best_output = None
min_reconstruction_loss = 0
for epoch in range(epochs):
    print(f"\n - ### Epoca {epoch+1} de {epochs}")
    for i in range(0, len(x_train), batch_size):
        print(f"\t - ## Batch {i+1} de {len(x_train)//batch_size}")
        x_batch = x_train[i:i+batch_size]
        loss, reconstruction_loss, kl_loss = train_step(x_batch)
        print("\t - TRAIN STEP [COMPLETE] ")
    if epoch == 0 or reconstruction_loss < min_reconstruction_loss:
        min_reconstruction_loss = reconstruction_loss
    print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")

# Salvando o melhor resultado em um arquivo txt

print("\n#################################################")
print("  Processo finalizado com sucesso! :) \n")
print("#################################################\n")
