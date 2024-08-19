
    # 1 -> 5s 9000Hz latent_dim = 2
    # 2 -> 7s 22050Hz latent_dim = 2

def reshape_batch(x_train, batch_size):
    x_train = x_train.reshape(batch_size, x_train.shape[1] // batch_size, x_train.shape[2]) # (1, 330000, 1) -> (33, 110000, 1)   33 Amostras de 110000 pontos
    return x_train