# Parâmetros do modelo
epochs = 200
num_mels = 512
batch_size = 1
latent_dim = 2
hidden_dims = [32, 64, 128, 256, 512]
num_samples_generate = 10
num_audio_samples = 8
kl_annealing_rate = 1 / epochs
max_kl_weight = 0.8

# Parâmetros de áudio
audio_duration = 7
audio_rate = 11000
audio_path = './audio/test/'

# Parâmetros do otimizador
learning_rate = 5e-3

