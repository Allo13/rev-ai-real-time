import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import soundfile as sf
import librosa

TARGET_SR = 16000  # XLSR-300M strictly requires 16 kHz input audio


class RawBoostAugmenter:
    """
    RawBoost Data Augmentation for raw audio waveforms.
    Simulates linear/non-linear convolutive noise, impulse response, 
    channel noise, and VoIP compression artifacts.
    """
    def __init__(self, algo=4, n_fft=1024, hop_length=256):
        self.algo = algo
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 1D waveform tensor of shape (samples,) or (1, samples)
        Returns:
            torch.Tensor: Augmented 1D waveform tensor of same shape
        """
        if self.algo == 0:
            return x

        is_2d = (x.ndim == 2)
        if is_2d:
            audio_np = x.squeeze(0).cpu().numpy()
        else:
            audio_np = x.cpu().numpy()

        if self.algo == 1:
            # Linear & Non-linear convolutive noise + additive noise
            aug_np = self._apply_linear_nonlinear_noise(audio_np)
        elif self.algo == 2:
            # Impulsive signal-dependent noise / FIR filtering
            aug_np = self._apply_impulsive_noise(audio_np)
        elif self.algo == 3:
            # Differential HV / VoIP compression simulation & bandpass filtering
            aug_np = self._apply_voip_channel_noise(audio_np)
        elif self.algo == 4:
            # Combined RawBoost augmentation (random selection among algos 1-3)
            chosen_algo = random.choice([1, 2, 3])
            if chosen_algo == 1:
                aug_np = self._apply_linear_nonlinear_noise(audio_np)
            elif chosen_algo == 2:
                aug_np = self._apply_impulsive_noise(audio_np)
            else:
                aug_np = self._apply_voip_channel_noise(audio_np)
        else:
            aug_np = audio_np

        aug_tensor = torch.from_numpy(aug_np.astype(np.float32))
        if is_2d:
            aug_tensor = aug_tensor.unsqueeze(0)
        return aug_tensor

    def _apply_linear_nonlinear_noise(self, audio: np.ndarray) -> np.ndarray:
        # Non-linear scaling + additive Gaussian noise
        alpha = random.uniform(0.8, 1.2)
        snr_db = random.uniform(10, 30)
        signal_power = np.mean(audio ** 2) + 1e-8
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
        # Non-linear harmonic distortion
        distorted = np.tanh(alpha * audio)
        return distorted + noise

    def _apply_impulsive_noise(self, audio: np.ndarray) -> np.ndarray:
        # FIR impulse filtering + sparse impulsive bursts
        fir_len = random.randint(5, 15)
        fir_filter = np.random.uniform(-0.5, 0.5, size=fir_len)
        fir_filter = fir_filter / np.sum(np.abs(fir_filter))
        filtered = np.convolve(audio, fir_filter, mode='same')
        
        # Add random impulses (simulating transmission glitches)
        num_impulses = random.randint(1, 10)
        impulse_indices = np.random.randint(0, len(audio), size=num_impulses)
        impulse_values = np.random.uniform(-0.5, 0.5, size=num_impulses)
        filtered[impulse_indices] += impulse_values
        return filtered

    def _apply_voip_channel_noise(self, audio: np.ndarray) -> np.ndarray:
        # VoIP codec simulation via bandpass filtering + amplitude quantization
        # Lowpass filter to simulate telephone 4kHz or G.711/G.722 codecs
        nyquist = TARGET_SR / 2.0
        cutoff = random.uniform(3000, 7500)
        btype_low = cutoff / nyquist
        
        # STFT domain spectral masking to simulate VoIP packet loss / compression
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # High-frequency attenuation (VoIP band-limiting)
        num_bins = magnitude.shape[0]
        cutoff_bin = int(num_bins * (cutoff / nyquist))
        magnitude[cutoff_bin:, :] *= random.uniform(0.01, 0.1)
        
        # Amplitude quantization simulation (mulaw-like quantization noise)
        bits = random.choice([6, 8, 10])
        quant_levels = 2 ** bits
        magnitude = np.round(magnitude * quant_levels) / quant_levels
        
        reconstructed = librosa.istft(magnitude * np.exp(1j * phase), hop_length=self.hop_length, length=len(audio))
        return reconstructed


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 / 2021 audio deepfake evaluation.
    Handles FLAC/WAV audio loading, strict 16 kHz resampling, fixed-length truncation/padding,
    and RawBoost augmentation.
    """
    def __init__(self, 
                 protocol_file: str = None, 
                 audio_dir: str = None, 
                 max_len: int = 64000, 
                 is_train: bool = True, 
                 rawboost_algo: int = 4,
                 num_synth_samples: int = 100):
        """
        Args:
            protocol_file (str): Path to ASVspoof protocol file (cm_protocols/...)
            audio_dir (str): Directory containing .flac audio files
            max_len (int): Number of audio samples per clip (64000 samples = 4s at 16kHz)
            is_train (bool): Enable augmentation during training
            rawboost_algo (int): RawBoost augmentation algorithm ID (0=None, 1..4)
            num_synth_samples (int): Fallback synthetic samples when data files aren't available
        """
        self.protocol_file = protocol_file
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.is_train = is_train
        self.rawboost = RawBoostAugmenter(algo=rawboost_algo if is_train else 0)
        
        self.file_list = []
        self.labels = []
        self.audio_ids = []

        if protocol_file and os.path.exists(protocol_file) and audio_dir and os.path.exists(audio_dir):
            self._load_protocol()
        else:
            # Fallback synthetic dataset generator for offline verification/benchmarking
            print(f"[ASVspoofDataset] Warning: protocol_file or audio_dir not found. Generating {num_synth_samples} synthetic samples for testing.")
            for i in range(num_synth_samples):
                self.file_list.append(f"synthetic_{i:05d}")
                # Alternate bonafide (0) and spoof (1)
                self.labels.append(i % 2)
                self.audio_ids.append(f"LA_E_{i:07d}")

    def _load_protocol(self):
        """Parses standard ASVspoof protocol text files."""
        with open(self.protocol_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Standard ASVspoof protocol format:
                # SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID - KEY
                audio_id = parts[1]
                key = parts[-1].lower()
                
                # Check for flac or wav file
                possible_path_flac = os.path.join(self.audio_dir, f"{audio_id}.flac")
                possible_path_wav = os.path.join(self.audio_dir, f"{audio_id}.wav")
                
                audio_path = None
                if os.path.exists(possible_path_flac):
                    audio_path = possible_path_flac
                elif os.path.exists(possible_path_wav):
                    audio_path = possible_path_wav
                else:
                    audio_path = possible_path_flac  # Default path target

                label = 0 if key == 'bonafide' else 1
                
                self.file_list.append(audio_path)
                self.labels.append(label)
                self.audio_ids.append(audio_id)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        audio_entry = self.file_list[idx]
        label = self.labels[idx]
        audio_id = self.audio_ids[idx]

        if isinstance(audio_entry, str) and audio_entry.startswith("synthetic_"):
            # Synthetic 16 kHz waveform generation
            t = np.linspace(0, self.max_len / TARGET_SR, self.max_len, endpoint=False)
            freq = 440.0 if label == 0 else 880.0
            waveform = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(self.max_len)
            waveform = waveform.astype(np.float32)
            sr = TARGET_SR
        else:
            try:
                # Load audio using soundfile or librosa
                waveform, sr = sf.read(audio_entry)
            except Exception:
                waveform, sr = librosa.load(audio_entry, sr=None)

            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)  # Convert stereo to mono

            # Strictly resample all audio to 16 kHz to match XLSR requirements
            if sr != TARGET_SR:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR

        # Apply padding or truncation to self.max_len
        num_samples = len(waveform)
        if num_samples < self.max_len:
            # Repeat waveform or zero-pad to max_len
            tile_count = (self.max_len // num_samples) + 1
            waveform = np.tile(waveform, tile_count)[:self.max_len]
        else:
            # Random crop during training, center crop during evaluation
            if self.is_train:
                max_start = num_samples - self.max_len
                start = random.randint(0, max_start)
                waveform = waveform[start:start + self.max_len]
            else:
                start = (num_samples - self.max_len) // 2
                waveform = waveform[start:start + self.max_len]

        waveform_tensor = torch.from_numpy(waveform).float()

        # Apply RawBoost data augmentation during training
        if self.is_train:
            waveform_tensor = self.rawboost(waveform_tensor)

        return waveform_tensor, label, audio_id
