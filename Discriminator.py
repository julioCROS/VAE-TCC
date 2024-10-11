import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)
