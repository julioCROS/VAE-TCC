import os
import random
import librosa
import numpy as np
from scipy.signal import firwin, lfilter, upfirdn

class AudioInfo:
    def __init__(self, audio_path, offset=0.0, duration=30, sr=44100):
        self.audio_path = audio_path
        self.offset = offset
        self.duration = duration
        self.sr = sr

    def _read_audio(self, path = None):
        y, sr = librosa.load(path, sr=self.sr, offset=self.offset, duration=self.duration)
        return y, sr
    
    def get_audio_data(self, num_audio_samples = 2):
        print("[Extraindo formas de onda]")
        files = [file for file in os.listdir(self.audio_path) if not file.startswith('.') and (file.endswith('.ogg') or file.endswith('.wav'))]
        files = random.sample(files, num_audio_samples)
        waveforms = []
        for file in files:
            ms, sr = self._read_audio(self.audio_path + file)
            waveforms.append(ms)
            print(f" - {file.rsplit('.', 1)[0]}: OK")
        waveforms = np.array(waveforms)
        print("")
        return waveforms

    def multiband_decomposition(self, data, num_bands=16):
      # Filtro prototípico (usando janela Kaiser)
      num_taps = 512  # Número de coeficientes do filtro
      cutoff = 1 / (2 * num_bands)  # Frequência de corte normalizada
      prototype_filter = firwin(num_taps, cutoff, window=('kaiser', 8.6))
      
      # Geração dos filtros modulados
      filters = []
      for k in range(num_bands):
          modulation = np.cos(2 * np.pi * k * (np.arange(num_taps) - (num_taps // 2)) / num_bands)
          band_filter = prototype_filter * modulation
          filters.append(band_filter)
      filters = np.array(filters)
      
      # Aplicação dos filtros e decimação
      subbands = []
      for band_filter in filters:
          # Filtragem do sinal
          filtered_signal = lfilter(band_filter, [1.0], data)
          # Decimação por fator igual ao número de bandas
          decimated_signal = filtered_signal[::num_bands]
          subbands.append(decimated_signal)
      
      return np.array(subbands)

    def multiband_synthesis(self, subbands, num_bands=16):
      num_taps = 512
      cutoff = 1 / (2 * num_bands)
      prototype_filter = firwin(num_taps, cutoff, window=('kaiser', 8.6))
      
      filters = []
      for k in range(num_bands):
          modulation = np.cos(2 * np.pi * k * (np.arange(num_taps) - (num_taps // 2)) / num_bands)
          band_filter = prototype_filter * modulation
          filters.append(band_filter)
      filters = np.array(filters)
      
      # Inicializar o sinal de saída
      audio = np.zeros(len(subbands[0]) * num_bands)
      
      for i, (band_filter, subband) in enumerate(zip(filters, subbands)):
          upsampled_signal = upfirdn([1], subband, up=num_bands)
          filtered_signal = lfilter(band_filter, [1.0], upsampled_signal)
          
          # Ajustar o comprimento de filtered_signal, se necessário
          if len(filtered_signal) < len(audio):
              filtered_signal = np.pad(filtered_signal, (0, len(audio) - len(filtered_signal)), mode='constant')
          elif len(filtered_signal) > len(audio):
              filtered_signal = filtered_signal[:len(audio)]
          
          audio += filtered_signal
        
      return audio