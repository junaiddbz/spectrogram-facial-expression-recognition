"""
make_spectrograms.py
─────────────────────
Phase 3: Spectrogram Generation Pipeline

Reads .npy signal files (output of extract_signals.py) and converts them
into 2D spectrogram images using Short-Time Fourier Transform (STFT).

Three signal groups are mapped to R, G, B channels of a single PNG image,
creating a compact multi-channel time-frequency representation.

Usage:
    python src/make_spectrograms.py

Output:
    Spectrogram PNG images organized by split and emotion:
    data/processed/spectrograms/{train,val}/{emotion}/{name}.png
"""

import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from tqdm import tqdm

from config import (
    CHANNEL_SIGNAL_MAP,
    IMG_HEIGHT,
    IMG_WIDTH,
    SIGNAL_NAMES,
    SIGNALS_DIR,
    SPECTROGRAMS_DIR,
    STFT_NFFT,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    VIDEO_FPS,
)

matplotlib.use("Agg")  # Non-interactive backend for server environments

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _compute_stft_magnitude(signal: np.ndarray) -> np.ndarray:
    """
    Compute the log-magnitude STFT of a 1D signal.

    Args:
        signal: 1D array of shape (SIGNAL_LENGTH,)

    Returns:
        log_mag: 2D array of shape (freq_bins, time_frames), values in dB
    """
    _, _, Zxx = stft(
        signal,
        fs=VIDEO_FPS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
    )
    magnitude = np.abs(Zxx)
    log_mag = 20.0 * np.log10(magnitude + 1e-8)  # Convert to dB, avoid log(0)
    return log_mag.astype(np.float32)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D float array to [0, 255] uint8."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255).astype(np.uint8)


def signals_to_spectrogram(signal_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of FAU signals into a multi-channel spectrogram image.

    Signal groups are averaged and assigned to R, G, B channels:
      - R channel: Mouth signals (lip aperture, mouth width)
      - G channel: Brow signals (left/right brow raise)
      - B channel: Eye/Jaw signals (eye openness, jaw open)

    Args:
        signal_matrix: np.ndarray of shape (NUM_SIGNALS, SIGNAL_LENGTH)

    Returns:
        img: RGB image of shape (IMG_HEIGHT, IMG_WIDTH, 3) as uint8
    """
    signal_dict = {name: signal_matrix[i] for i, name in enumerate(SIGNAL_NAMES)}

    channels = []
    for _channel_label, signal_names_in_channel in CHANNEL_SIGNAL_MAP.items():
        # Average the STFT magnitudes of all signals in this channel group
        stft_mags = [
            _compute_stft_magnitude(signal_dict[n]) for n in signal_names_in_channel
        ]
        avg_mag = np.mean(stft_mags, axis=0)  # (freq_bins, time_frames)
        channel_img = _normalize_to_uint8(avg_mag)

        # Resize to target image size
        resized = cv2.resize(
            channel_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        channels.append(resized)

    # Stack into RGB image (H, W, 3)
    rgb_image = np.stack(channels, axis=2)
    return rgb_image


def process_all_signals(signals_dir: Path, output_dir: Path) -> None:
    """
    Read the signals_metadata.csv and generate spectrogram PNGs
    for all signal files.

    Args:
        signals_dir:  Directory containing .npy files + signals_metadata.csv
        output_dir:   Root output directory for spectrogram images
    """
    csv_path = signals_dir / "signals_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}. " "Run extract_signals.py first."
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} signal entries from {csv_path}")

    success, failed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating spectrograms"):
        signal_path = Path(row["signal_path"])
        emotion = row["emotion"]
        split = row["split"]

        if not signal_path.exists():
            log.warning(f"Signal file not found: {signal_path}. Skipping.")
            failed += 1
            continue

        signal_matrix = np.load(signal_path)  # (NUM_SIGNALS, SIGNAL_LENGTH)

        try:
            spectrogram_img = signals_to_spectrogram(signal_matrix)
        except Exception as e:
            log.warning(f"Failed to generate spectrogram for {signal_path.name}: {e}")
            failed += 1
            continue

        # Save to: output_dir/{split}/{emotion}/{stem}.png
        out_dir = output_dir / split / emotion
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (signal_path.stem + ".png")
        cv2.imwrite(str(out_path), cv2.cvtColor(spectrogram_img, cv2.COLOR_RGB2BGR))
        success += 1

    log.info(f"Done. ✓ {success} spectrograms saved, ✗ {failed} failed.")


def visualize_pipeline(
    signal_matrix: np.ndarray, title: str = "Pipeline Visualization"
) -> plt.Figure:
    """
    Generate a multi-panel figure showing:
    1. The raw FAU time-series signals
    2. The individual channel STFT spectrograms
    3. The final composite RGB spectrogram image

    Useful for the notebooks and README figures.

    Args:
        signal_matrix: np.ndarray of shape (NUM_SIGNALS, SIGNAL_LENGTH)
        title:         Figure title

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(18, 10), facecolor="#0f172a")
    fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=0.98)

    n_signals = len(SIGNAL_NAMES)
    gs = fig.add_gridspec(3, n_signals, hspace=0.4, wspace=0.3)

    signal_dict = {name: signal_matrix[i] for i, name in enumerate(SIGNAL_NAMES)}

    # Row 1: Raw signals
    for i, (name, signal) in enumerate(signal_dict.items()):
        ax = fig.add_subplot(gs[0, i])
        t = np.linspace(0, len(signal) / VIDEO_FPS, len(signal))
        ax.plot(t, signal, color="#a78bfa", linewidth=1.5)
        ax.set_title(name.replace("_", "\n"), color="white", fontsize=8)
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Row 2: Per-signal STFT spectrograms
    for i, (name, signal) in enumerate(signal_dict.items()):
        ax = fig.add_subplot(gs[1, i])
        log_mag = _compute_stft_magnitude(signal)
        ax.imshow(log_mag, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"STFT: {name.replace('_', ' ')}", color="white", fontsize=7)
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="gray", labelsize=6)

    # Row 3: Final composite RGB image
    ax_rgb = fig.add_subplot(gs[2, n_signals // 2 - 1 : n_signals // 2 + 2])
    composite = signals_to_spectrogram(signal_matrix)
    ax_rgb.imshow(composite)
    ax_rgb.set_title(
        "Composite RGB Spectrogram (Input to CNN)", color="white", fontsize=10
    )
    ax_rgb.axis("off")

    return fig


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_all_signals(SIGNALS_DIR, SPECTROGRAMS_DIR)
