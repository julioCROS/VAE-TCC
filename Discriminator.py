import tensorflow as tf
from tensorflow.keras import layers
import config

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        discr_layers = []

        for h_dim in config.discr_hidden_dims:
            discr_layers.append(layers.Conv2D(h_dim, config.discr_kernel_size, strides=config.discr_strides, padding='same'))
            discr_layers.append(layers.LeakyReLU())
        discr_layers.append(layers.GlobalAveragePooling2D())
        discr_layers.append(layers.Dense(1))
        
        self.model = tf.keras.Sequential(discr_layers)

    def call(self, inputs):
        return self.model(inputs)
