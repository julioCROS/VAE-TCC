import time
import tensorflow as tf
import soundfile as sf

from VAE import VAE, Discriminator
from utils import *
from AudioData import AudioData
import config 

start_time = time.time()
current_id = generate_random_id()
print(f"[[ ID do Experimento ]] - {current_id}")

# Carregando os dados de áudio
audio_data = AudioData(audio_path=config.audio_path, duration=config.audio_duration, sr=config.audio_rate, n_mels=config.num_mels)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)  

# Extraindo dados de Treinamento
data = audio_data.get_mel_spectrograms(num_audio_samples=config.num_audio_samples) 
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)          
print(f" [[ Dados de Treinamento ]] - Shape: {data.shape}")  

# Inicializando o modelo
model = VAE(input_shape=data.shape, latent_dim=config.latent_dim, hidden_dims=config.hidden_dims, id=current_id, duration=config.audio_duration, rate=config.audio_rate, kl_annealing_rate=config.kl_annealing_rate, max_kl_weight=config.max_kl_weight)

# Treinando a representação do modelo
model.representation_learning_train(data, config.epochs, optimizer)

# Treinando o modelo
best_epoch_metadata, reconstruction_losses, kl_losses = model.train(data, config.epochs, optimizer)

# Treinando o modelo com adversarial fine-tuning
#discriminator = Discriminator()
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#model.adversarial_fine_tuning_train(data, config.epochs, optimizer, discriminator_optimizer, discriminator)

# Avaliando espaço latente
#model.latent_space_compactness(data)

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Obtendo medias MU para visualização do espaço latente
mu = model.encode(data)[0]
mu = mu.numpy()

# Exibindo os resultados
show_results(best_epoch_metadata, execution_time, reconstruction_losses, kl_losses, config.epochs, current_id, mu)
save_metadata('stg', current_id, config.audio_path, config.audio_duration, config.audio_rate, config.latent_dim, config.batch_size, config.epochs, best_epoch_metadata, execution_time, fad=None, mels=config.num_mels)

# Salvando o modelo de VAE treinado
model.save('./models/vae_' + current_id + '.h5')

# Gerando N espectrogramas a partir do espaço latente
generated = model.sample(config.num_samples_generate)

# Para cada espectrograma gerado, salva o resultado em formato de áudio e em um arquivo txt
for i in range(config.num_samples_generate):
    curr_gen = generated[i]
    file_result = './results/generated_audio_' + current_id + '_' + str(i+1) + '.ogg'
    generated_audio = audio_data.mel_spectrogram_to_audio(curr_gen.numpy().reshape(curr_gen.shape[0], curr_gen.shape[1]))
    print(f"[ Salvando resultado gerado {i + 1} ] - {current_id}")
    sf.write(file_result, generated_audio, config.audio_rate)