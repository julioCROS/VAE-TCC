import librosa
import numpy as np

class AudioInfo:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def read_audio(self, offset = 0.0, duration = 30, sr=None, n_mels = 512, n_fft = 2048):
        y, sr = librosa.load(self.audio_path, sr=sr, offset=offset, duration=duration)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db)
        self.y = y
        self.sr = sr
        self.mel_spectrogram_db = mel_spectrogram_db
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        return mel_spectrogram_db, sr

    def get_duration(self):
        return librosa.get_duration(y=self.y, sr=self.sr)
    
    def spectrogram_to_audio(self, mel_spectrogram_db):
        mel_spectrogram_db = mel_spectrogram_db * np.std(mel_spectrogram_db) + np.mean(mel_spectrogram_db)
        mel_spectrogram = librosa.db_to_power(mel_spectrogram_db)
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)
        power_spectrogram = np.dot(mel_basis.T, mel_spectrogram)
        magnitude_spectrogram = power_spectrogram
        return librosa.griffinlim(magnitude_spectrogram, n_fft=self.n_fft, hop_length=self.hop_length)


    