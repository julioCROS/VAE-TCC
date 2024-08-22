import librosa
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

def calculate_fad(input, output):
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

def mel_spectrogram_db_to_audio(mel_spectrogram_db, sr, n_mels, n_fft=2048, hop_length=512):
    mel_spectrogram_db = mel_spectrogram_db * np.std(mel_spectrogram_db) + np.mean(mel_spectrogram_db)
    mel_spectrogram = librosa.db_to_power(mel_spectrogram_db)
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    power_spectrogram = np.dot(mel_basis.T, mel_spectrogram)
    magnitude_spectrogram = power_spectrogram
    audio = librosa.griffinlim(magnitude_spectrogram, n_fft=n_fft, hop_length=hop_length)
    
    return audio

def show_results(output, execution_time, fad):
    print("#" * 112)
    print("\nMelhor resultado: ")
    print(f"\t- Epoca: {output[1] + 1}")
    print(f"\t- FAD: {fad}")
    print(f"\t- Loss: {output[2].numpy()}")
    print(f"\t- Reconstrução Loss: {output[3].numpy()}")
    print(f"\t- KL Loss: {output[4].numpy()}")
    print(f"\t- Tempo de execução: {execution_time} segundos")

def save_metadata(type, id, path, duration, rate, latent_dim, batch_size, epochs, output, execution_time, fad, mels = None):
    metadata_file = None
    if type is None:
        metadata_file = './results/results_metadata.txt'
    elif type == 'stg':
        metadata_file = './results/results_spectrogram_metadata.txt'
    with open(metadata_file, 'a') as file:
        file.write(f"ID: {id}\n")
        file.write(f"Audio Path: {path}\n")
        file.write(f"Audio Duration: {duration}\n")
        file.write(f"Audio Rate: {rate}\n")
        file.write(f"Latent Dim: {latent_dim}\n")
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Epochs: {epochs}\n")
        if mels is not None:
            file.write(f"Num. Mels: {mels}\n")
        file.write(f"Best Epoch: {output[1]+1}\n")
        file.write(f"FAD: {fad}\n")
        file.write(f"Loss: {output[2].numpy()}\n")
        file.write(f"Reconstruction Loss: {output[3].numpy()}\n")
        file.write(f"KL Loss: {output[4].numpy()}\n")
        file.write(f"Execution Time: {execution_time} seconds\n")
        file.write("_" * 50 + "\n")