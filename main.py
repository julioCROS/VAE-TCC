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
epochs = 2                                      # Número de épocas
num_batches = 1                                 # Tamanho do lote
latent_dim = 128                                # Dimensão do espaço latente
hidden_dims = [32, 64, 128, 256]                # Dimensões ocultas
num_mels = 512                                  # Qntde. de mels
audio_duration = 5; audio_rate = 11000          # Duração e taxa de amostragem do áudio
audio_path = './audio/ethereal.ogg'             # Caminho do arquivo de áudio
audio_data = AudioInfo(audio_path)              # Leitor de áudio
optimizer = tf.keras.optimizers.Adam()          # Otimizador

# Dados de Treinamento                                                                                      # Valores usados como referencia: 30s, 11000Hz, 512 mels, 1 batch
train_data, sample_rate = audio_data.read_audio(duration=audio_duration, sr=audio_rate, n_mels=num_mels)    # (512, 645)
train_data = get_batches(train_data, num_batches)                                                           # (1, 512, 645)
train_data = np.expand_dims(train_data, axis=-1)                                                            # (1, 512, 645, 1) Dimensão extra para indicar o numero de canais (1)
original_audio = audio_data.spectrogram_to_audio(train_data.reshape(train_data.shape[1] * num_batches, train_data.shape[2]))
save_result('STG_input', original_audio, current_id)                                                        # Salvando os dados de entrada em um arquivo txt

# Instanciando o modelo VAE
print("\n [ Instanciando o modelo VAE - " + current_id + " ]")
vae = VAE(input_shape=train_data.shape, latent_dim=latent_dim, id = current_id, duration=audio_duration, rate=audio_rate, hidden_dims=hidden_dims)
vae.summary()
vae.encoder.summary()
vae.decoder.summary()

# Treinando o modelo VAE
print("\n[ Iniciando Treinamento ]")
output = train_vae(vae, train_data, epochs, optimizer)     
output_audio = audio_data.spectrogram_to_audio(output[0].numpy().reshape(output[0].shape[1], output[0].shape[2]))
save_result('STG_output', output_audio, current_id)
print("[ Treinamento Concluído ]")

# Calculando FAD a partir da entrada e da melhor saída
fad = None
original_spectrogram = train_data.reshape(train_data.shape[1], train_data.shape[2])
output_spectrogram = output[0].numpy().reshape(train_data.shape[1], train_data.shape[2])
try:
    print("[ Calculando FAD... ]")
    fad = calculate_fad(original_spectrogram, output_spectrogram)
except Exception as e:
    print(f"\nErro ao calcular FAD: {e}")
    print(f"Input shape: {original_spectrogram.shape} | Output shape: {output_spectrogram.shape}")
    fad = '-'

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Exibindo os resultados
show_results(output, execution_time, fad)

# Salvando os metadados em um arquivo txt
save_metadata("stg", current_id, audio_path, audio_duration, audio_rate, latent_dim, batch_size, epochs, output, execution_time, fad, num_mels)

# Salvando o modelo de VAE treinado
vae.save('./models/vae_' + current_id + '.h5')

# Gerando 5 espectrogramas a partir do espaço latente
generated = vae.sample(5)

# Para cada espectrograma gerado, salva o resultado em formato de áudio e em um arquivo txt
for i in range(5):
    print(f"[ Salvando resultado gerado {i + 1} ] | Generated shape: {generated[i].shape}")
    generated_audio = audio_data.spectrogram_to_audio(generated[i].numpy().reshape(generated[i].shape[0], generated[i].shape[1]))
    save_result('STG_generated_' + str(i), generated_audio, current_id)





