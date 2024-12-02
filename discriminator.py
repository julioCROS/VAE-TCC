import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self, hidden_dims, kernel_sizes, strides):
        super(Discriminator, self).__init__()
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Definindo as camadas convolucionais
        self.conv_layers = [
            layers.Conv1D(h_dim, kernel_size=k, strides=s, padding="same")
            for h_dim, k, s in zip(self.hidden_dims, self.kernel_sizes, self.strides)
        ]
        self.activation = layers.LeakyReLU()  # Ativação LeakyReLU
        self.global_avg_pooling = layers.GlobalAveragePooling1D()  # Pooling global
        self.output_layer = layers.Dense(1)  # Camada densa para logits finais

    def call(self, inputs, return_features=False):
        """
        Argumentos:
        - inputs: Tensor de entrada.
        - return_features: Booleano indicando se deve retornar as features intermediárias.

        Retorna:
        - Se `return_features` for True: (logits, features)
        - Caso contrário: logits
        """
        x = inputs
        features = []  # Lista para armazenar as features intermediárias

        # Passa os dados pelas camadas convolucionais
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
            features.append(x)  # Armazena a ativação intermediária

        # Global Average Pooling
        x = self.global_avg_pooling(x)

        # Camada de saída
        logits = self.output_layer(x)

        # Retorna features intermediárias, se solicitado
        if return_features:
            return logits, features
        return logits