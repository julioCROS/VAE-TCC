# Parâmetros do modelo
epochs = 100
num_mels = 512
latent_dim = 16
hidden_dims = [8, 16, 32, 64, 128]
compact_latent_space = False
kl_annealing_rate = 1 / epochs
max_kl_weight = 0.5
kernel_sizes = [(7, 7), (7, 7), (4, 4)]
strides = [(3, 3), (3, 3), (2, 2)]
num_samples_generate = 2

# Parâmetros de áudio
num_audio_samples = 2
num_audio_segments  = 2
shuffle_segments = True
audio_duration = 16
audio_rate = 11000
audio_path = './audio/test/'

# Parâmetros do otimizador
learning_rate = 5e-4

