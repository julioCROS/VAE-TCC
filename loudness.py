import tensorflow as tf
import numpy as np
from scipy.signal import windows
from librosa.core import fft_frequencies, A_weighting

class Loudness(tf.keras.layers.Layer):
    def __init__(self, sr, block_size, n_fft=2048):
        """
        Inicializa a camada de Loudness.

        Args:
            sr (int): Taxa de amostragem (sample rate).
            block_size (int): Tamanho do salto (hop size).
            n_fft (int): Tamanho da FFT.
        """
        super(Loudness, self).__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft
        
        # Frequências FFT
        f = fft_frequencies(sr=sr, n_fft=n_fft) + 1e-7
        
        # Curva A-Weighting
        self.a_weight = tf.convert_to_tensor(A_weighting(f), dtype=tf.float32)
        
        # Janela de Hann
        hann_window = windows.hann(n_fft, sym=False)
        self.window = tf.convert_to_tensor(hann_window, dtype=tf.float32)

    def call(self, x):
        """
        Calcula a loudness ponderada A para um sinal de áudio.

        Args:
            x (tf.Tensor): Tensor de entrada com formato (batch_size, num_samples).
        
        Returns:
            tf.Tensor: Tensor da loudness calculada com formato (batch_size, 1, 1).
        """
        
        # STFT (Short-Time Fourier Transform)
        stft_result = tf.signal.stft(
            x,
            frame_length=self.n_fft,
            frame_step=self.block_size,
            fft_length=self.n_fft,
            window_fn=lambda: self.window,
            pad_end=True
        )
        
        # Magnitude do espectro
        magnitude = tf.abs(stft_result)
        
        # Aplicar escala logarítmica e curva A-weighting
        log_magnitude = tf.math.log(magnitude + 1e-7)
        weighted_magnitude = log_magnitude + self.a_weight
        
        # Média ao longo do eixo de frequência
        loudness = tf.reduce_mean(weighted_magnitude, axis=1, keepdims=True)
        return loudness
