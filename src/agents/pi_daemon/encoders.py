from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class VisionEncoder(Protocol):
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a 1D float32 array of length 256 (vision embedding)."""


class AudioEncoder(Protocol):
    def encode_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 1D float32 array of length 256 (auditory embedding)."""


class _RandomProjector:
    def __init__(self, input_dim: int, output_dim: int = 256, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.matrix = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
        self.scale = 1.0 / math.sqrt(input_dim)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        return (vector.astype(np.float32) @ self.matrix * self.scale).astype(np.float32)


class DefaultVisionEncoder:
    def __init__(self, target_size: int = 32, seed: int = 42):
        self.target_size = target_size
        input_dim = target_size * target_size
        self.projector = _RandomProjector(input_dim, 256, seed)

    @staticmethod
    def _downsample_grayscale(frame: np.ndarray, target_size: int) -> np.ndarray:
        # Convert to grayscale
        gray = frame.astype(np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        h, w = gray.shape
        # Select evenly spaced indices to downsample
        y_idx = np.linspace(0, h - 1, target_size).astype(int)
        x_idx = np.linspace(0, w - 1, target_size).astype(int)
        downsampled = gray[np.ix_(y_idx, x_idx)]
        return downsampled

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        downsampled = self._downsample_grayscale(frame, self.target_size)
        normalized = (downsampled - downsampled.mean()) / (downsampled.std() + 1e-6)
        flat = normalized.flatten()
        return self.projector(flat)


class DefaultAudioEncoder:
    def __init__(self, n_mel_bins: int = 64, n_frames: int = 32, seed: int = 1234):
        self.n_mel_bins = n_mel_bins
        self.n_frames = n_frames
        input_dim = n_mel_bins * n_frames
        self.projector = _RandomProjector(input_dim, 256, seed)

    def _spectrogram(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = waveform.astype(np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        hop_length = max(1, len(waveform) // self.n_frames)
        window = 512
        specs = []
        for start in range(0, len(waveform), hop_length):
            end = start + window
            if end > len(waveform):
                break
            segment = waveform[start:end] * np.hanning(window)
            spectrum = np.abs(np.fft.rfft(segment))
            specs.append(spectrum)
            if len(specs) >= self.n_frames:
                break
        if not specs:
            specs = [np.zeros(window // 2 + 1, dtype=np.float32)]
        spec_matrix = np.stack(specs, axis=1)
        # Simple binning to approximate mel bands
        freq_bins = spec_matrix.shape[0]
        mel = np.zeros((self.n_mel_bins, spec_matrix.shape[1]), dtype=np.float32)
        edges = np.linspace(0, freq_bins, self.n_mel_bins + 1).astype(int)
        for i in range(self.n_mel_bins):
            start, end = edges[i], edges[i + 1]
            if end <= start:
                end = min(start + 1, freq_bins)
            mel[i] = spec_matrix[start:end].mean(axis=0)
        mel = np.log1p(mel)
        if mel.shape[1] < self.n_frames:
            pad_width = self.n_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="edge")
        elif mel.shape[1] > self.n_frames:
            mel = mel[:, : self.n_frames]
        return mel

    def encode_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Encode a mel-like matrix of shape (n_mel_bins, n_frames) into a 256-d vector.
        Used for both waveform-derived spectrograms and spectrogram images.
        """
        mel = mel.astype(np.float32)
        # Resize to (n_mel_bins, n_frames) by simple subsampling
        h, w = mel.shape
        y_idx = np.linspace(0, h - 1, self.n_mel_bins).astype(int)
        x_idx = np.linspace(0, w - 1, self.n_frames).astype(int)
        mel_resized = mel[np.ix_(y_idx, x_idx)]
        flat = mel_resized.flatten()
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)
        return self.projector(flat)

    def encode_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        mel_spec = self._spectrogram(waveform, sample_rate)
        return self.encode_mel(mel_spec)


def build_default_vision_encoder() -> VisionEncoder:
    return DefaultVisionEncoder()


def build_default_audio_encoder() -> AudioEncoder:
    return DefaultAudioEncoder()
