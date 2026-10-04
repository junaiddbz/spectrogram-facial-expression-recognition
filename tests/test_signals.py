"""
test_signals.py
───────────────
Unit tests for the signal extraction pipeline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import NUM_SIGNALS, SIGNAL_LENGTH, SIGNAL_NAMES  # noqa: E402
from make_spectrograms import (  # noqa: E402
    _compute_stft_magnitude,
    _normalize_to_uint8,
    signals_to_spectrogram,
)


class TestSTFT:
    """Tests for STFT-based spectrogram generation."""

    def test_stft_magnitude_shape(self):
        """STFT should return a 2D array."""
        signal = np.random.randn(SIGNAL_LENGTH).astype(np.float32)
        mag = _compute_stft_magnitude(signal)
        assert mag.ndim == 2, "STFT output should be 2D (freq_bins × time_frames)"

    def test_stft_no_inf_or_nan(self):
        """Log-magnitude STFT should not contain inf or nan."""
        signal = np.random.randn(SIGNAL_LENGTH).astype(np.float32)
        mag = _compute_stft_magnitude(signal)
        assert not np.any(np.isnan(mag)), "STFT output contains NaN"
        assert not np.any(np.isinf(mag)), "STFT output contains Inf"

    def test_stft_flat_signal(self):
        """STFT of a flat (zero) signal should not crash."""
        signal = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
        mag = _compute_stft_magnitude(signal)
        assert mag.ndim == 2


class TestNormalization:
    """Tests for image normalization utility."""

    def test_output_range(self):
        arr = np.random.randn(16, 16).astype(np.float32)
        result = _normalize_to_uint8(arr)
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8

    def test_constant_array(self):
        """Constant array should produce all-zero output without error."""
        arr = np.full((16, 16), 3.14, dtype=np.float32)
        result = _normalize_to_uint8(arr)
        assert np.all(result == 0)


class TestSpectrogramGeneration:
    """Tests for the composite spectrogram image generation."""

    def _dummy_matrix(self) -> np.ndarray:
        return np.random.randn(NUM_SIGNALS, SIGNAL_LENGTH).astype(np.float32)

    def test_output_shape(self):
        """Spectrogram image should be (IMG_HEIGHT, IMG_WIDTH, 3)."""
        from config import IMG_HEIGHT, IMG_WIDTH

        matrix = self._dummy_matrix()
        img = signals_to_spectrogram(matrix)
        assert img.shape == (
            IMG_HEIGHT,
            IMG_WIDTH,
            3,
        ), f"Expected ({IMG_HEIGHT}, {IMG_WIDTH}, 3), got {img.shape}"

    def test_output_dtype(self):
        """Spectrogram image should be uint8."""
        img = signals_to_spectrogram(self._dummy_matrix())
        assert img.dtype == np.uint8

    def test_output_range(self):
        """Pixel values should be in [0, 255]."""
        img = signals_to_spectrogram(self._dummy_matrix())
        assert img.min() >= 0
        assert img.max() <= 255

    def test_signal_names_match_config(self):
        """Number of signal names in config must match NUM_SIGNALS."""
        assert (
            len(SIGNAL_NAMES) == NUM_SIGNALS
        ), f"SIGNAL_NAMES length {len(SIGNAL_NAMES)} != NUM_SIGNALS {NUM_SIGNALS}"
