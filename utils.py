import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm

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

def get_batches(data, num_batches):
    dim_1 = data.shape[1] // (num_batches // data.shape[0])
    if data.shape[1] % num_batches == 0:
        data = data.reshape(num_batches, dim_1, data.shape[2], data.shape[3])
    else:
        raise ValueError("A divisão não resulta em dimensões inteiras")
    print(f" - Dados separados em {num_batches} lotes de tamanho {dim_1}")
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

def show_results(metadata, execution_time, reconstruction_losses, kl_losses, epochs, id, mu):
    print("#" * 112)
    print("\nMelhor resultado: ")
    print(f"\t- Epoca: {metadata[0] + 1}")
    print(f"\t- Loss: {metadata[1].numpy()}")
    print(f"\t- Reconstrução Loss: {metadata[2].numpy()}")
    print(f"\t- KL Loss: {metadata[3].numpy()}")
    print(f"\t- Tempo de execução: {execution_time} segundos\n")

    plt.plot(range(1, epochs+1), reconstruction_losses, label='Reconstrução')
    plt.plot(range(1, epochs+1), kl_losses, label='Perda KL')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig('./graphs/graph_' + id + '.png')
    plt.close() 

    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu)
    plt.scatter(mu_pca[:, 0], mu_pca[:, 1])
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Visualização do Espaço Latente com PCA')
    plt.savefig('./graphs/pca_' + id + '.png')
    plt.close() 

def save_metadata(type, id, path, duration, rate, latent_dim, batch_size, epochs, output, execution_time, fad, mels = None):
    metadata_file = None
    if type is None:
        metadata_file = './results/results_metadata.txt'
    elif type == 'stg':
        metadata_file = './results/results_spectrogram_metadata.txt'
    with open(metadata_file, 'a') as file:
        print(f"[Salvando metadados de {id}]")
        file.write(f"ID: {id}\n")
        file.write(f"Audio Path: {path}\n")
        file.write(f"Audio Duration: {duration}\n")
        file.write(f"Audio Rate: {rate}\n")
        file.write(f"Latent Dim: {latent_dim}\n")
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Epochs: {epochs}\n")
        if mels is not None:
            file.write(f"Num. Mels: {mels}\n")
        file.write(f"Best Epoch: {output[0]+1}\n")
        file.write(f"KL Weight: {output[4]}\n")
        file.write(f"FAD: {fad}\n")
        file.write(f"Loss: {output[1].numpy()}\n")
        file.write(f"Reconstruction Loss: {output[2].numpy()}\n")
        file.write(f"KL Loss: {output[3].numpy()}\n")
        file.write(f"Execution Time: {execution_time} seconds\n")
        file.write("_" * 50 + "\n")
        print(f"[Metadados salvos em {metadata_file}]\n")