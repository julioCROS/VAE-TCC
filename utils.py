import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
import configs

def calculate_fad(input, output):
    input = input.reshape(input.shape[0], -1)  
    output = output.reshape(output.shape[0], -1)

    mu_input = np.mean(input, axis=0)
    sigma_input = np.cov(input, rowvar=False)

    mu_output = np.mean(output, axis=0)
    sigma_output = np.cov(output, rowvar=False)

    mu_diff = mu_input - mu_output
    covmean, _ = sqrtm(sigma_input.dot(sigma_output), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = np.sum(mu_diff**2) + np.trace(sigma_input + sigma_output - 2 * covmean)
    return fad

def generate_random_id():
    return ''.join([chr(np.random.randint(65, 91)) for _ in range(2)]) + str(np.random.randint(1000, 9999))

def save_result(type, data, id, dim_1 = None, dim_2 = None):
    if dim_1 is None or dim_2 is None:
        np.savetxt('./results/' + type + '_' +  id + '.txt' , data, fmt = "%f")
    else:
        np.savetxt('./results/' + type + '_' +  id + '.txt' , data.reshape(dim_1 * dim_2, ), fmt = "%f")

def save_intermediate_output(id, epoch, output, duration, rate):
    np.savetxt('./results/intermediate_output_' +  id + '_' + str(epoch) + '.txt', output.numpy().reshape(duration * rate, ), fmt = "%f")

def remove_intermediate_outputs(id):
    import os
    for file in os.listdir('./results'):
        if file.startswith('intermediate_output_' + id):
            os.remove('./results/' + file)

def show_results(execution_time):
    print("[TREINAMENTOS CONCLUIDOS]")
    print(f"\t - Tempo de execução: {execution_time} segundos")

def save_metadata(id, execution_time):
    metadata_file = '/content/drive/MyDrive/VAE_TCC/results/results_spectrogram_metadata.txt'
    with open(metadata_file, 'a') as file:
        print(f"[Salvando metadados de {id}]")
        file.write(f"ID: {id}\n")
        file.write(f"Epochs: {config.epochs}\n")
        file.write(f"Num. audio samples: {config.num_audio_samples}\n")
        file.write(f"Num. Mels: {config.num_mels}\n")
        file.write(f"Latent Dim: {config.latent_dim}\n")
        file.write(f"Compact Latent Space: {config.compact_latent_space}\n")
        file.write(f"Audio Duration: {config.audio_duration}\n")
        file.write(f"Audio Rate: {config.audio_rate}\n")
        file.write(f"Hidden Dims: {config.hidden_dims}\n")
        file.write(f"KL Annealing Rate: {config.kl_annealing_rate}\n")
        file.write(f"Max KL Weight: {config.max_kl_weight}\n")
        file.write(f"Optimizer Learning Rate: {config.learning_rate}\n")
        file.write(f"Num. Segments: {config.num_audio_segments}\n")
        file.write(f"Shuffle Segments: {config.shuffle_segments}\n")
        file.write(f"Kernel Sizes: {config.kernel_sizes}\n")
        file.write(f"Strides: {config.strides}\n")
        file.write(f"Execution Time: {execution_time} seconds\n")
        file.write("_" * 50 + "\n")
        print(f"[Metadados salvos em {metadata_file}]")

def save_graphs(id, signal_losses, representation_train_kl_losses, gen_lossess, discr_lossess, mu, compact_latent_space = False, reduced_latent = None):
    representation_train_kl_losses = representation_train_kl_losses[10:]
    graph_file = '/content/drive/MyDrive/VAE_TCC/graphs/' + id + '.png'
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].plot(signal_losses)
    axs[0].set_title("Signal Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(representation_train_kl_losses)
    axs[1].set_title("KL Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")

    axs[2].plot(gen_lossess)
    axs[2].set_title("Gen. Loss")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")

    axs[3].plot(discr_lossess)
    axs[3].set_title("Discr. Loss")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")


    '''
    pca = PCA(n_components=3)
    mu_pca = pca.fit_transform(mu)
    axs[2].scatter(mu_pca[:, 0], mu_pca[:, 1], c='blue', marker='o')
    axs[2].set_title("PCA Latent Vectors [ORIGINAL]")
    axs[2].set_xlabel("Principal Component 1")
    axs[2].set_ylabel("Principal Component 2")

    if compact_latent_space is True:
      if reduced_latent.shape[1] >= 2:
        axs[3].scatter(reduced_latent[:, 0], reduced_latent[:, 1], cmap='viridis', s=30)
        axs[3].set_xlabel("Dim. 1")
        axs[3].set_ylabel("Dim. 2")
      else:
        axs[3].scatter(reduced_latent[:, 0], [0] * len(reduced_latent), cmap='viridis', s=30)
        axs[3].set_xlabel("Dim. 1")
        axs[3].set_ylabel("Constant")

      axs[3].set_title("[COMPACT] Latent Space")
    '''

    plt.tight_layout()
    plt.savefig(graph_file)
    plt.close() 
    print(f"[Graficos salvos em {graph_file}]\n")