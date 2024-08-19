import numpy as np
import soundfile as sf

def generate_random_id():
    return ''.join([chr(np.random.randint(65, 91)) for _ in range(2)]) + str(np.random.randint(1000, 9999))

def save_result(type, data, id, duration, rate):
    np.savetxt('./results/' + type + '_' +  id + '.txt' , data.reshape(duration * rate, ), fmt = "%f")

def save_intermediate_output(id, epoch, output, duration, rate):
    np.savetxt('./results/intermediate_output_' +  id + '_' + str(epoch) + '.txt', output.numpy().reshape(duration * rate, ), fmt = "%f")

def remove_intermediate_outputs(id):
    import os
    for file in os.listdir('./results'):
        if file.startswith('intermediate_output_' + id):
            os.remove('./results/' + file)

def show_results(best_output, execution_time):
    print("#" * 112)
    print("Melhor resultado: ")
    print(f"\t- Epoca: {best_output[1] + 1}")
    print(f"\t- Loss: {best_output[2].numpy()}")
    print(f"\t- Reconstrução Loss: {best_output[3].numpy()}")
    print(f"\t- KL Loss: {best_output[4].numpy()}")
    print(f"\t- Tempo de execução: {execution_time} segundos")

def save_metadata(id, path, duration, rate, latent_dim, batch_size, epochs, best_output, execution_time):
    with open('./results/results_metadata.txt', 'a') as file:
        file.write(f"ID: {id}\n")
        file.write(f"Audio Path: {path}\n")
        file.write(f"Audio Duration: {duration}\n")
        file.write(f"Audio Rate: {rate}\n")
        file.write(f"Latent Dim: {latent_dim}\n")
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write(f"Best Epoch: {best_output[1]+1}\n")
        file.write(f"Loss: {best_output[2].numpy()}\n")
        file.write(f"Reconstruction Loss: {best_output[3].numpy()}\n")
        file.write(f"KL Loss: {best_output[4].numpy()}\n")
        file.write(f"Execution Time: {execution_time} seconds\n")
        file.write("_" * 50 + "\n")