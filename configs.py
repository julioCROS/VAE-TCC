# Parâmetros do modelo VAE
EPOCHS = 1
DELTA = 0.1
KL_BETA = 0.1
PATIENCE = 30
EPOCH_PRINT = 2
BATCH_SIZE = -1
LOUD_STRIDE = 1
LATENT_DIM = 128
MIN_KL_LOSS = 0.1
USE_NOISE = False
RESIDUAL_DEPTH = 3
VAE_STRIDES = [4, 4, 4, 2]
VAE_KERNELS = [8, 8, 8, 4]
COMPACT_LATENT_SPACE = True
VAE_HIDDEN_DIMS = [64, 128, 256, 512]

# Parametros do modelo Discriminator
DISCR_EPOCHS = EPOCHS
DISCR_STRIDES = [1, 4, 4, 4]
DISCR_KERNELS = [15, 15, 15, 15]
DISCR_HIDDEN_DIMS = [64, 128, 256, 512]

# Parâmetros de áudio
NUM_BANDS = 16
AUDIO_RATE = 44000
AUDIO_DURATION = 10
NUM_AUDIO_SAMPLES = 1
NUM_SAMPLES_GENERATE = 1
AUDIO_PATH = '/content/drive/MyDrive/VAE_TCC/audio/test/'

# Parâmetros do otimizador
BETA_1 = 0.5
BETA_2 = 0.9
EPSILON = 1e-7
LEARNING_RATE = 1e-4
