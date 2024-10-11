# Parâmetros do modelo VAE
epochs = 1
num_mels = 512
latent_dim = 8
hidden_dims = [16, 32, 64, 128]
compact_latent_space = False
kl_annealing_rate = 1 / epochs
max_kl_weight = 0.8
kernel_sizes = [(6, 6), (6, 6), (3, 3)]
strides = [(3, 3), (3, 3), (2, 2)]
num_samples_generate = 2

# Parametros do modelo Discriminator
discr_hidden_dims = [64, 128, 256, 512]
discr_kernel_size = (5,1)
discr_strides = (3,1)

# Parâmetros de áudio
num_audio_samples = 2
num_audio_segments  = 2
shuffle_segments = False
audio_duration = 16
audio_rate = 11000
audio_path = './audio/test/'

# Parâmetros do otimizador
learning_rate = 5e-3

