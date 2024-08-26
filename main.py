import time
import numpy as np
import tensorflow as tf

from VAE import VAE
from utils import *
from train_funcs import *
from audio_info import AudioInfo

start_time = time.time()
current_id = generate_random_id()

# Parâmetros
epochs = 2                                               # Número de épocas
num_mels = 512                                           # Qntde. de mels
num_batches = 1                                          # Tamanho do lote
latent_dim = 128                                         # Dimensão do espaço latente
hidden_dims = [32, 64, 128, 256]                         # Dimensões ocultas

audio_duration = 5; audio_rate = 11000                   # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'                      # Caminho do arquivo de áudio
audio_data = AudioInfo(audio_path)                       # Leitor de áudio
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # Otimizador

# Dados de Treinamento                                                                                      # Valores usados como referencia: 30s, 11000Hz, 512 mels, 1 batch
train_data, sample_rate = audio_data.read_audio(duration=audio_duration, sr=audio_rate, n_mels=num_mels)    # (512, 645)
train_data = get_batches(train_data, num_batches)                                                           # (1, 512, 645)
train_data = np.expand_dims(train_data, axis=-1)                                                            # (1, 512, 645, 1) Dimensão extra para indicar o numero de canais (1)

# Instanciando o modelo VAE
print("\n [ Instanciando o modelo VAE - " + current_id + " ]")
vae = VAE(input_shape=train_data.shape, latent_dim=latent_dim, id = current_id, duration=audio_duration, rate=audio_rate, hidden_dims=hidden_dims)

# Treinando o modelo VAE
print("\n[ Iniciando Treinamento ]")
train_metadata = train_vae(vae, train_data, epochs, optimizer)     

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Exibindo os resultados
show_results(train_metadata, execution_time)

# Salvando o modelo de VAE treinado
vae.save('./models/vae_' + current_id + '.h5')

# Gerando 5 espectrogramas a partir do espaço latente
generated = vae.sample(5)

# Para cada espectrograma gerado, salva o resultado em formato de áudio e em um arquivo txt
for i in range(5):
    print(f"[ Salvando resultado gerado {i + 1} ]")
    generated_audio = audio_data.spectrogram_to_audio(generated[i].numpy().reshape(generated[i].shape[0], generated[i].shape[1]))
    save_result('STG_generated_' + str(i), generated_audio, current_id)

print(f"\n [ Conversão concluída] | Espectrograma {generated[0].shape} - Áudio {generated_audio.shape}")





