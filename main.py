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
epochs = 10                                    # Número de épocas
batch_size = '-'                               # Tamanho do lote
latent_dim = 2                                 # Dimensão do espaço latente
num_mels = 512                                 # Qntde. de mels
audio_duration = 5; audio_rate = 11000        # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'            # Caminho do arquivo de áudio
audio_data = AudioInfo(audio_path)             # Leitor de áudio
optimizer = tf.keras.optimizers.Adam()         # Otimizador

# Dados de Treinamento                                                                                      # Valores usados como referencia: 30s, 11000Hz, 512 mels
train_data, sample_rate = audio_data.read_audio(duration=audio_duration, sr=audio_rate, n_mels=num_mels)    # (512, 645)
train_data = np.expand_dims(train_data, axis=0)                                                             # (1, 512, 645) Dimensão extra para indicar que é 1 lote de áudio completo com formato (512, 645)
train_data = np.expand_dims(train_data, axis=-1)                                                            # (1, 512, 645, 1) Dimensão extra para indicar o numero de canais (1)
print("Data shape: ", train_data.shape)
save_result('stg_input', train_data, current_id, train_data.shape[1], train_data.shape[2])                  # Salvando os dados de entrada em um arquivo txt
train_data_audio = audio_data.spectrogram_to_audio(train_data.reshape(train_data.shape[1], train_data.shape[2])) 
print("Data audio shape: ", train_data_audio.shape)
save_result('stg_input_AUDIO', train_data_audio, current_id)                  # Salvando os dados de entrada em um arquivo txt


# Instanciando o modelo VAE
vae = VAE(input_shape=train_data.shape, latent_dim=latent_dim, id = current_id, duration=audio_duration, rate=audio_rate)
#vae.summary()
#vae.encoder.summary()
#vae.decoder.summary()

# Treinando o modelo VAE
#output = train_vae(vae, train_data, epochs, optimizer)     
print("\n[ Treinamento Concluído ]")



