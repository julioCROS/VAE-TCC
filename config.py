# Parâmetros do modelo
epochs = 10
num_mels = 512
batch_size = 1
latent_dim = 8
hidden_dims = [64, 128, 256, 512, 1024]
num_samples_generate = 10
num_audio_samples = 2
kl_annealing_rate = 1 / epochs
max_kl_weight = 2

# Parâmetros de áudio
audio_duration = 15
audio_rate = 11000
audio_path = './audio/etc/'

# Parâmetros do otimizador
learning_rate = 5e-3

