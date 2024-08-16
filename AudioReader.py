import librosa

class AudioReader:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def read_audio(self, offset = 0.0, duration = 30, sr=None):
        y, sr = librosa.load(self.audio_path, sr=sr, offset=offset, duration=duration)
        self.y = y
        self.sr = sr
        return y, sr

    def get_duration(self):
        return librosa.get_duration(y=self.y, sr=self.sr)