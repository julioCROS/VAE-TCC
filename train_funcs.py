from utils import save_intermediate_output
import tensorflow as tf

@tf.function
def train_step(vae, data, optimizer):
    with tf.GradientTape() as tape:
        outputs, inputs, mu, log_var = vae(data)
        loss, reconstruction_loss, kl_loss, output = vae.loss_function(inputs, outputs, mu, log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss, reconstruction_loss, kl_loss, output

def train_vae(vae, data, epochs, optimizer):
    best_output = None
    min_reconstruction_loss = float('inf')
    for epoch in range(epochs):
        loss, reconstruction_loss, kl_loss, output = train_step(vae, data, optimizer) 
        if reconstruction_loss < min_reconstruction_loss:
            min_reconstruction_loss = reconstruction_loss
            best_output = output, epoch, loss, reconstruction_loss, kl_loss
        print(f"Epoca {epoch+1} | Loss: {loss.numpy()} |  Recon. Loss: {reconstruction_loss.numpy()} | KL Loss: {kl_loss.numpy()}")
        if epoch % 10 == 0:
            save_intermediate_output(vae.id, epoch, output, vae.duration, vae.rate)
    return best_output