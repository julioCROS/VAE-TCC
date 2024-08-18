import numpy as np
import tensorflow as tf
from VAE_2 import VAE

x_train = np.random.rand(4000, 1)                # Dados de treinamento
batch_size = 2000                                  # Tamanho do lote
latent_dim = 128                                   # Dimensão do espaço latente

# Inicialização do Modelo
vae = VAE(input_shape = x_train.shape, latent_dim = latent_dim)
vae.summary()
