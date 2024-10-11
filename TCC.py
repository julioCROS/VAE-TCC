import time
import tensorflow as tf
import soundfile as sf

from VAE import VAE
from utils import *
from AudioData import AudioData
import config 

start_time = time.time()
current_id = generate_random_id()
print(f"\n\n[[ ID do Experimento ]] - {current_id}\n ")

# Carregando os dados de áudio
audio_data = AudioData(audio_path=config.audio_path, duration=config.audio_duration, sr=config.audio_rate, n_mels=config.num_mels)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)  

# Extraindo dados de Treinamento
data = audio_data.get_mel_spectrograms(num_audio_samples=config.num_audio_samples) 
print(f"[ Dados de Treinamento ] - Shape: {data.shape}")  
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1) 
data = get_segments(data, config.num_audio_segments, config.shuffle_segments)
print(f"[ Dados de Treinamento segmentados ] - Shape: {data.shape} | Shuffle: {config.shuffle_segments}\n")  

# Inicializando o modelo
model = VAE(input_shape=data.shape, latent_dim=config.latent_dim, hidden_dims=config.hidden_dims, 
            id=current_id, duration=config.audio_duration, rate=config.audio_rate, kernel_sizes=config.kernel_sizes, 
            strides=config.strides, kl_annealing_rate=config.kl_annealing_rate, max_kl_weight=config.max_kl_weight)


# Treinando a representação do modelo
spectral_losses, representation_train_kl_losses = model.representation_learning_train(data, config.epochs, optimizer)

# Treinando o modelo com adversarial fine-tuning
#discriminator = Discriminator()
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#model.adversarial_fine_tuning_train(data, config.epochs, optimizer, discriminator_optimizer, discriminator)

# Avaliando espaço latente
#reduced_latent, informative_dimensions = model.compact_latent_representation(data)

# Calculando o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
execution_time = str(round(execution_time, 2))

# Obtendo medias MU para visualização do espaço latente
mu = model.encode(data)[0]
mu = mu.numpy()

# Exibindo e salvando resultados
show_results(execution_time)
save_metadata(current_id, execution_time)
save_graphs(current_id, spectral_losses, representation_train_kl_losses, mu)

# Salvando o modelo de VAE treinado
model.save('./models/vae_' + current_id + '.h5')

# Gerando N espectrogramas a partir do espaço latente
generated = model.sample(config.num_samples_generate, data, None, config.compact_latent_space)

# Para cada espectrograma gerado, salva o resultado em formato de áudio e em um arquivo txt
for i in range(config.num_samples_generate):
    curr_gen = generated[i]
    file_result = './results/generated_audio_' + current_id + '_' + str(i+1) + '.ogg'
    generated_audio = audio_data.mel_spectrogram_to_audio(curr_gen.numpy().reshape(curr_gen.shape[0], curr_gen.shape[1]))
    print(f"[ Salvando resultado gerado {i + 1} ] - {current_id}")
    sf.write(file_result, generated_audio, config.audio_rate)