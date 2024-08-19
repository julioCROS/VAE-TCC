import time
import numpy as np
import tensorflow as tf

from VAE import VAE
from AudioReader import AudioReader

start_time = time.time()

# Parâmetros
audio_duration = 7; audio_rate = 22500       # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'           # Caminho do arquivo de áudio
audio_data = AudioReader(audio_path)          # Leitor de áudio
latent_dim = 16                               # Dimensão do espaço latente
batch_size = 1                                # Tamanho do lote
epochs = 50                                   # Número de épocas

# Dados de Treinamento                                                                      # Valores usados como referencia: 30s a 11000Hz = 330000
x_train, sample_rate = audio_data.read_audio(duration= audio_duration, sr=audio_rate)       # (330000,)
x_train = x_train.reshape(x_train.shape[0], 1)                                              # (330000, 1) Dimensão extra para indicar que o áudio tem apenas um canal
x_train = np.expand_dims(x_train, axis=0)                                                   # (1, 330000, 1) Dimensão extra para indicar que é 1 lote de áudios completo

# Salvando os dados de treinamento em um arquivo txt
def generate_random_id():
    return ''.join([chr(np.random.randint(65, 91)) for _ in range(2)]) + str(np.random.randint(1000, 9999))

current_id = generate_random_id()

np.savetxt('./files/input' +  current_id + '.txt' , x_train.reshape(audio_duration * audio_rate, ), fmt = "%f")

# Inicialização do Modelo
vae = VAE(input_shape = x_train.shape, latent_dim = latent_dim)
vae.summary()
vae.encoder.summary()
vae.decoder.summary()

# Treinamento do Modelo
optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        outputs, inputs, mu, log_var = vae(x)
        loss, reconstruction_loss, kl_loss, output = vae.loss_function(inputs, outputs, mu, log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss, reconstruction_loss, kl_loss, output

# Loop de Treinamento
best_output = None
min_reconstruction_loss = float('inf')
for epoch in range(epochs):
    loss, reconstruction_loss, kl_loss, output = train_step(x_train) 
    if reconstruction_loss < min_reconstruction_loss:
        min_reconstruction_loss = reconstruction_loss
        best_output = output, epoch, loss, reconstruction_loss, kl_loss
    print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")

end_time = time.time()
execution_time = end_time - start_time

print("################################################################################################################" )
print("Melhor resultado: ")
print(f"\t- Epoca: {best_output[1]+1}")
print(f"\t- Loss: {best_output[2].numpy()}")
print(f"\t- Reconstrução Loss: {best_output[3].numpy()}")
print(f"\t- KL Loss: {best_output[4].numpy()}")
print(f"\t- Tempo de execução: {execution_time} segundos")

# Salvando os dados de saída em um arquivo txt
np.savetxt('./files/output_' +  current_id + '.txt', best_output[0].numpy().reshape(audio_duration * audio_rate, ), fmt = "%f")

# Salvando os metadados em um arquivo txt
def save_metadata():
    with open('./metadata_audio.txt', 'w') as file:
        file.write(f"ID: {current_id}\n")
        file.write(f"Audio Path: {audio_path}\n")
        file.write(f"Audio Duration: {audio_duration}\n")
        file.write(f"Audio Rate: {audio_rate}\n")
        file.write(f"Latent Dim: {latent_dim}\n")
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write(f"Best Epoch: {best_output[1]+1}\n")
        file.write(f"Loss: {best_output[2].numpy()}\n")
        file.write(f"Reconstruction Loss: {best_output[3].numpy()}\n")
        file.write(f"KL Loss: {best_output[4].numpy()}\n")
        file.write(f"Execution Time: {execution_time} seconds\n")
        file.write(f"Sample Rate: {sample_rate}\n")
        file.write("_" * 50 + "\n")

save_metadata()