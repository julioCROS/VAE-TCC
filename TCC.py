import numpy as np
import tensorflow as tf
from VAE import VAE

# Parâmetros
dim_1 = 100; dim_2 = 100; dim_3 = 1     # Dimensões dos dados de entrada
input_shape = (dim_1, dim_2, dim_3)     # Simulando os pontos de dados do áudio
latent_dim = 256                        # Dimensão do espaço latente
batch_size = 100                        # Tamanho do lote
epochs = 100                            # Número de épocas

# Dados de Treinamento
# x_train = np.random.randint(1, 20, (dim_1, dim_2, dim_3)) # Simulando com dados inteiros
x_train = np.random.rand(dim_1, dim_2, dim_3)               # Simulando com dados reais

# Salvando os dados de treinamento em um arquivo txt
np.savetxt('./files/input_FLOAT.txt', x_train.reshape(dim_1, dim_2), fmt='%f')

# Convertendo x_train para tensor
x_train = tf.convert_to_tensor(x_train)
# Inicialização do Modelo
vae = VAE(input_shape = input_shape, latent_dim = latent_dim)

# Compilação do Modelo
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        outputs, inputs, mu, log_var = vae(x)
        loss, reconstruction_loss, kl_loss, output = vae.loss_function(inputs, outputs, mu, log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss, reconstruction_loss, kl_loss, output

# Loop de Treinamento
best_output = None
min_reconstruction_loss = 0
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        loss, reconstruction_loss, kl_loss, output = train_step(x_batch)
    if epoch == 0 or reconstruction_loss < min_reconstruction_loss:
        min_reconstruction_loss = reconstruction_loss
        best_output = output, reconstruction_loss, kl_loss, loss, epoch
    print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")

# Salvando o melhor resultado em um arquivo txt
np.savetxt('./files/output_FLOAT.txt', best_output[0].numpy().reshape(dim_1, dim_2), fmt='%f')

print("\n#################################################")
print("  Processo finalizado com sucesso! :) \n")
print(" - Melhor Reconstrução Loss:", best_output[1].numpy())
print(" - KL Loss:", best_output[2].numpy())
print(" - Loss:", best_output[3].numpy())
print(" - Época:", best_output[4])
print("#################################################\n")
