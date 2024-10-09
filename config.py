# Parâmetros do modelo
epochs = 100
num_mels = 512
latent_dim = 2
hidden_dims = [16, 32, 64, 128, 512]
compact_latent_space = False
kl_annealing_rate = 1 / epochs
max_kl_weight = 0.8
kernel_sizes = [(5, 5), (5, 5), (3, 3)]
strides = [(3, 3), (3, 3), (2, 2)]
num_samples_generate = 2

# Parâmetros de áudio
num_audio_samples = 2
num_audio_segments  = 32
shuffle_segments = True
audio_duration = 7
audio_rate = 11000
audio_path = './audio/test/'

# Parâmetros do otimizador
learning_rate = 5e-3

