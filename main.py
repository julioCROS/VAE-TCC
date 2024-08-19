import time
import numpy as np
import tensorflow as tf

from vae import VAE
from utils import *
from train_funcs import *
from audio_info import AudioInfo

start_time = time.time()
current_id = generate_random_id()

# Parâmetros
audio_duration = 5; audio_rate = 22050         # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'            # Caminho do arquivo de áudio
audio_data = AudioInfo(audio_path)             # Leitor de áudio
latent_dim = 128                               # Dimensão do espaço latente
batch_size = '-'                               # Tamanho do lote
epochs = 100                                   # Número de épocas
optimizer = tf.keras.optimizers.Adam()         # Otimizador

# Dados de Treinamento                                                                      # Valores usados como referencia: 30s a 11000Hz = 330000
x_train, sample_rate = audio_data.read_audio(duration= audio_duration, sr=audio_rate)       # (330000,)
x_train = x_train.reshape(x_train.shape[0], 1)                                              # (330000, 1) Dimensão extra para indicar que o áudio tem apenas um canal
x_train = np.expand_dims(x_train, axis=0)                                                   # (1, 330000, 1) Dimensão extra para indicar que é 1 lote de áudios completo
save_result('input', x_train, current_id, audio_duration, audio_rate)                       # Salvando os dados de entrada em um arquivo txt

# Instanciando o modelo VAE
vae = VAE(input_shape = x_train.shape, latent_dim = latent_dim, id = current_id, duration = audio_duration, rate = audio_rate)                             

# Treinando o modelo VAE
best_output = train_vae(vae, x_train, epochs, optimizer)                                    
save_result('output', best_output[0].numpy(), current_id, audio_duration, audio_rate)       # Salvando os dados da melhor saída em um arquivo txt
remove_intermediate_outputs(current_id)                                                     # Removendo os arquivos de saída intermediários
print("\n[ Treinamento Concluído ]")

# Calculando FAD a partir da entrada e da melhor saída
fad = None
try:
    print("[ Calculando FAD... ]\n")
    fad = calculate_fad(x_train.reshape(audio_duration, audio_rate), best_output[0].numpy().reshape(audio_duration, audio_rate))
except Exception as e:
    print(f"Erro ao calcular FAD: {e}")
    fad = None

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Exibindo os resultados
show_results(best_output, execution_time, fad)

# Salvando os metadados em um arquivo txt
save_metadata(current_id, audio_path, audio_duration, audio_rate, latent_dim, batch_size, epochs, best_output, execution_time, fad)