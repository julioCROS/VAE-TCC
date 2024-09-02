import os
import random
import librosa
import numpy as np


class AudioData:
    def __init__(self, audio_path, offset=0.0, duration=30, sr=None, n_mels=512, n_fft=2048):
        self.audio_path = audio_path
        self.offset = offset
        self.duration = duration
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

    def read_audio(self, path = None):
        y, sr = librosa.load(path, sr=self.sr, offset=self.offset, duration=self.duration)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft)
        return mel_spectrogram, sr
    
    def get_mel_spectrograms(self, num_samples = 2):
        print("\n[ Extraindo Mel Spectrograms... ]")
        files = os.listdir(self.audio_path)
        files = random.sample(files, 2)
        mel_spectrograms = []
        for file in files:
            ms, sr = self.read_audio(self.audio_path + file)
            mel_spectrograms.append(ms)
            print(f" - {file.split('.')[0]} OK")
        print("\n")
        mel_spectrograms = np.array(mel_spectrograms)
        return mel_spectrograms
    
    def mel_spectrogram_to_audio(self, mel_spectrogram):
        audio_data = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        return audio_data


    