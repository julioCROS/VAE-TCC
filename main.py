import time
import numpy as np
import tensorflow as tf

from VAE import VAE
from utils import *
from AudioData import AudioData

start_time = time.time()
current_id = generate_random_id()

# Parâmetros
epochs = 600                                             # Número de épocas
num_mels = 512                                           # Qntde. de mels
batch_size = 1                                           # Tamanho do lote
latent_dim = 256                                         # Dimensão do espaço latente
hidden_dims = [32, 64, 128, 256]                         # Dimensões ocultas

audio_duration = 10; audio_rate = 11000                   # Duração e taxa de amostragem do áudio
audio_path = './audio/'                                  # Caminho do arquivo de áudio

# Carregando os dados de áudio
audio_data = AudioData(audio_path = audio_path, duration = audio_duration, sr = audio_rate, n_mels = num_mels) 
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # Otimizador

# Extraindo dados de Treinamento                                                                            # Valores usados como referencia: 30s, 11000Hz, 512 mels, e 1 arquivo com 1 batch
data = audio_data.get_mel_spectrograms(num_samples=2)                                                       # (1, 512, 645)
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)                                         # (1, 512, 645, 1) Adicionando a dimensão do canal

# Inicializando o modelo
model = VAE(input_shape=data.shape, latent_dim=latent_dim, hidden_dims=hidden_dims, id=current_id, duration=audio_duration, rate=audio_rate)

# Treinando o modelo
best_epoch_metadata = model.train(data, epochs, optimizer)

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Exibindo os resultados
show_results(best_epoch_metadata, execution_time)
save_metadata('stg', current_id, './audio/', audio_duration, audio_rate, latent_dim, batch_size, epochs, best_epoch_metadata, execution_time, fad = None, mels = num_mels)

# Salvando o modelo de VAE treinado
model.save('./models/vae_' + current_id + '.h5')

# Gerando 5 espectrogramas a partir do espaço latente
generated = model.sample(5)

# Para cada espectrograma gerado, salva o resultado em formato de áudio e em um arquivo txt
for i in range(7):
    print(f"[ Salvando resultado gerado {i + 1} ]")
    generated_audio = audio_data.mel_spectrogram_to_audio(generated[i].numpy().reshape(generated[i].shape[0], generated[i].shape[1]))
    save_result('generated_audio', generated_audio, current_id + '_' + str(i), audio_duration * audio_rate)


