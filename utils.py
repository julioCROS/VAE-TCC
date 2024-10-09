import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
import config 

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

def get_segments(data, num_audio_segments, shuffle_segments):
    if num_audio_segments <= data.shape[0]:
        return data
    
    dim_1 = data.shape[1] // (num_audio_segments // data.shape[0])

    if data.shape[1] % num_audio_segments == 0:
        data = data.reshape(num_audio_segments, dim_1, data.shape[2], data.shape[3])
    else:
        raise ValueError("A divisão não resulta em dimensões inteiras ]")
    
    print(f"[ Dados separados em {num_audio_segments} segmentos de tamanho {dim_1}]")

    if shuffle_segments is True:
        np.random.shuffle(data)

    return data

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
    print("#" * 112)
    print("[TREINAMENTOS CONCLUIDOS]")
    print(f"Tempo de execução: {execution_time} segundos\n")
    print("#" * 112 + '\n')

def save_metadata(id, execution_time):
    metadata_file = './results/results_spectrogram_metadata.txt'
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
        print(f"[Metadados salvos em {metadata_file}]\n")

def save_graphs(id, train_recon_losses, train_kl_losses, mu, spectral_losses = None, representation_train_kl_losses = None,):
    #representation_train_kl_losses = representation_train_kl_losses[10:]
    #train_kl_losses = train_kl_losses[10:]

    graph_file = './graphs/' + id + '.png'
    
    ''' 
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    axs[0, 0].plot(spectral_losses)
    axs[0, 0].set_title("Spectral Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(representation_train_kl_losses)
    axs[0, 1].set_title("Representation Train KL Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    '''

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    axs[0].plot(train_recon_losses)
    axs[0].set_title("Train Recon Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(train_kl_losses)
    axs[1].set_title("Train KL Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")

    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu)
    axs[2].scatter(mu_pca[:, 0], mu_pca[:, 1], c='blue', marker='o')
    axs[2].set_title("PCA Latent Vectors")
    axs[2].set_xlabel("Principal Component 1")
    axs[2].set_ylabel("Principal Component 2")

  
    plt.tight_layout()

    plt.savefig(graph_file)
    plt.close() 
    print(f"[Graficos salvos em {graph_file}]\n")


