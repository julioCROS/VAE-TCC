import os
import random
import librosa
import numpy as np
import torch
import torch.nn.functional as F
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
        # Gera o filtro protótipo
        num_taps = 512
        cutoff = 1 / (2 * num_bands)
        prototype_filter = firwin(num_taps, cutoff, window=('kaiser', 8.6))
        h = torch.from_numpy(prototype_filter).float()

        # Gera os filtros modulados
        hk = self._get_qmf_bank(h, num_bands)
        hk = self._center_pad_next_pow_2(hk)  # Garante tamanho adequado

        # Realiza convolução polifásica
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # B x 1 x T
        subbands = self._polyphase_forward(data_tensor, hk)
        return subbands.squeeze(0).numpy()

    def multiband_synthesis(self, subbands, num_bands=16):
        """
        Função de síntese multibanda para reconstruir o sinal completo.
        
        Args:
            subbands (np.ndarray): Sub-bandas decodificadas (num_bands x T_decimated).
            num_bands (int): Número de bandas.

        Returns:
            np.ndarray: Sinal sintetizado reconstruído.
        """
        # Gera o filtro protótipo
        num_taps = 512
        cutoff = 1 / (2 * num_bands)
        prototype_filter = firwin(num_taps, cutoff, window=('kaiser', 8.6))
        h = torch.from_numpy(prototype_filter).float()

        # Gera os filtros modulados
        hk = self._get_qmf_bank(h, num_bands)
        hk = self._center_pad_next_pow_2(hk)  # Garante tamanho correto para convoluções

        # Converte sub-bandas para tensor (B x num_bands x T_decimated)
        subbands_tensor = torch.from_numpy(subbands).unsqueeze(0)

        # Realiza a síntese polifásica
        reconstructed_signal = self._polyphase_inverse(subbands_tensor, hk)

        # Retorna o sinal reconstruído como numpy array
        return reconstructed_signal.squeeze(0).numpy()
    
    def _get_qmf_bank(self, h, num_bands):
        """
        Gera o banco de filtros QMF (Quadrature Mirror Filter) a partir do filtro protótipo.
        
        Args:
            h (torch.Tensor): Filtro protótipo (1D tensor).
            num_bands (int): Número de bandas (filtros).

        Returns:
            torch.Tensor: Banco de filtros QMF (num_bands x len(h)).
        """
        num_taps = h.shape[0]
        filters = []
        
        for k in range(num_bands):
            # Modulação para criar o filtro da banda k
            modulation = torch.cos(2 * torch.pi * k * (torch.arange(num_taps) - (num_taps // 2)) / num_bands)
            band_filter = h * modulation
            filters.append(band_filter)
        
        return torch.stack(filters)  # Retorna um tensor de shape (num_bands, num_taps)
    

    
    def _polyphase_forward(self, audio, hk):
        """
        Realiza a decomposição polifásica para dividir o sinal em sub-bandas.
        
        Args:
            audio (torch.Tensor): Sinal de entrada (B x 1 x T).
            hk (torch.Tensor): Banco de filtros QMF (num_bands x len(h)).
        
        Returns:
            torch.Tensor: Tensor de sub-bandas (B x num_bands x T_decimated).
        """
        B, _, T = audio.shape  # O áudio deve ter 3 dimensões
        num_bands, num_taps = hk.shape
        T_decimated = (T + num_taps - 1) // num_bands  # Comprimento das sub-bandas
        
        subbands = torch.zeros(B, num_bands, T_decimated, device=audio.device)
        
        for band in range(num_bands):
            # Adicionar dimensões ao kernel
            kernel = hk[band].unsqueeze(0).unsqueeze(1)  # (1, 1, kernel_size)
            print(f"Kernel {band} shape: {kernel.shape}")
            
            # Convolução polifásica
            filtered_signal = F.conv1d(audio, kernel, padding=num_taps - 1)
            print(f"Filtered signal shape (band {band}): {filtered_signal.shape}")
            print(f"T_decimated: {T_decimated}")
            print(f"Filtered signal (after downsampling): {filtered_signal[:, ::num_bands].shape}")
           

            # Subamostragem com truncamento
            filtered_signal = filtered_signal.squeeze(1)  # Remove canal redundante
            print(f"Filtered signal (after truncation): {filtered_signal[:, ::num_bands][:, :T_decimated].shape}")
            subbands[:, band, :] = filtered_signal[:, ::num_bands][:, :T_decimated]
        
        return subbands
    
    def _polyphase_inverse(self, subbands, hk):
        """
        Realiza a síntese polifásica para reconstrução do sinal.
        
        Args:
            subbands (torch.Tensor): Tensor de sub-bandas (B x num_bands x T_decimated).
            hk (torch.Tensor): Banco de filtros QMF (num_bands x len(h)).
        
        Returns:
            torch.Tensor: Sinal reconstruído (B x T_reconstructed).
        """
        B, num_bands, T_decimated = subbands.shape
        num_taps = hk.shape[1]
        T_reconstructed = T_decimated * num_bands

        # Inicializa o sinal de saída
        reconstructed_signal = torch.zeros(B, T_reconstructed + num_taps - 1)

        for band in range(num_bands):
            # Upsample: Interpola o sinal (adiciona zeros entre as amostras)
            upsampled_band = F.pad(subbands[:, band, :].unsqueeze(1), (0, T_decimated * (num_bands - 1)))
            upsampled_band = upsampled_band.view(B, 1, -1)  # B x 1 x T_upsampled
            
            # Filtragem: Convolução do sinal interpolado com o filtro da banda
            filtered_band = F.conv1d(upsampled_band, hk[band].unsqueeze(0).unsqueeze(1))
            
            # Soma no sinal final
            reconstructed_signal[:, :filtered_band.shape[-1]] += filtered_band.squeeze(1)
        
        return reconstructed_signal[:, :T_reconstructed]  # Retorna no tamanho correto
    
    def _center_pad_next_pow_2(self, x):
        """
        Padroniza o comprimento para o próximo tamanho que é potência de 2.
        
        Args:
            x (torch.Tensor): Tensor 1D ou 2D (C x T ou T).
        
        Returns:
            torch.Tensor: Tensor com comprimento ajustado.
        """
        original_len = x.shape[-1]
        next_pow_2 = 2 ** (original_len - 1).bit_length()
        pad_total = next_pow_2 - original_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        if x.ndim == 1:
            return F.pad(x, (pad_left, pad_right), mode='constant', value=0)
        elif x.ndim == 2:
            return F.pad(x, (pad_left, pad_right), mode='constant', value=0)
        else:
            raise ValueError("Input tensor must be 1D or 2D.")