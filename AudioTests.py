import numpy as np
from AudioReader import AudioReader

audio_path = './audio/ethereal.ogg'
audio_data = AudioReader(audio_path)

sample_, sample_rate = audio_data.read_audio()
print(" - Audio rate:", sample_rate)
print(" - Audio shape:", sample_.shape)
print(" - Duration:", audio_data.get_duration())
print(" - Audio data:", sample_)

# # Salvando os dados de áudio em um arquivo txt
np.savetxt('./files/input_AUDIO.txt', sample_.reshape(30, 44100), fmt = "%f")
