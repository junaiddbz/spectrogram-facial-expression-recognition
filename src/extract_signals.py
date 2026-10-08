"""
extract_signals.py
──────────────────
Phase 2: Facial Landmark Signal Extraction Pipeline

This script processes video files (CK+ or RAVDESS) frame-by-frame using
MediaPipe Face Mesh. For each frame, it computes Euclidean distances between
specific landmark pairs (Facial Action Units), producing 1D time-series signals
that encode facial motion over time.

Usage:
    python src/extract_signals.py --data_dir data/raw/CK+ --output_dir data/processed/signals

Output:
    One .npy file per video clip: shape (NUM_SIGNALS, SIGNAL_LENGTH)
    One metadata CSV: signals_metadata.csv (path, emotion label, split)
"""

import argparse
import csv
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    LANDMARK_SIGNALS,
    NUM_SIGNALS,
    RANDOM_SEED,
    SIGNAL_LENGTH,
    SIGNAL_NAMES,
    SIGNALS_DIR,
    TRAIN_RATIO,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D landmark points."""
    return float(np.linalg.norm(p1 - p2))


def extract_signals_from_video(video_path: Path) -> np.ndarray | None:
    """
    Extract per-frame Facial Action Unit (FAU) distance signals from a video.

    Args:
        video_path: Path to the input video file.

    Returns:
        signals: np.ndarray of shape (NUM_SIGNALS, SIGNAL_LENGTH),
                 or None if face detection fails on too many frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return None

    signals = {name: [] for name in SIGNAL_NAMES}
    failed_frames = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 478 landmarks (iris + lips precision)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe requires RGB input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                # Face not detected — propagate last value.
                # If no face has been detected yet, do not append 0.0 (prevents STFT impulse distortion)  # noqa: E501
                failed_frames += 1
                if signals[SIGNAL_NAMES[0]]:
                    for name in SIGNAL_NAMES:
                        signals[name].append(signals[name][-1])
                continue

            lm = results.multi_face_landmarks[0].landmark

            # Convert normalized [0,1] coords to pixel coords
            pts = np.array([[lm_point.x * w, lm_point.y * h, lm_point.z * w] for lm_point in lm])

            for name, (idx_a, idx_b) in LANDMARK_SIGNALS.items():
                dist = _euclidean(pts[idx_a], pts[idx_b])
                # Normalize by inter-ocular distance for scale invariance
                inter_ocular = _euclidean(pts[33], pts[263]) + 1e-6
                signals[name].append(dist / inter_ocular)

    cap.release()

    # Quality check: reject if >40% frames failed
    total_frames = sum(len(v) for v in signals.values()) // NUM_SIGNALS
    if total_frames == 0 or failed_frames / max(total_frames, 1) > 0.4:
        log.warning(f"Too many failed frames ({failed_frames}) in {video_path.name}. Skipping.")
        return None

    # Resample each signal to fixed SIGNAL_LENGTH using linear interpolation
    signal_matrix = []
    for name in SIGNAL_NAMES:
        raw = np.array(signals[name], dtype=np.float32)
        resampled = np.interp(
            np.linspace(0, len(raw) - 1, SIGNAL_LENGTH),
            np.arange(len(raw)),
            raw,
        )
        signal_matrix.append(resampled)

    return np.array(signal_matrix, dtype=np.float32)  # (NUM_SIGNALS, SIGNAL_LENGTH)


def process_dataset(data_dir: Path, output_dir: Path) -> None:
    """
    Walk a dataset directory, extract signals from all video clips,
    and save results as .npy files with a metadata CSV.

    Expected directory structure (CK+ style):
        data_dir/
          happy/
            S001_happy_01.avi
            ...
          sad/
            ...

    Args:
        data_dir:   Root directory containing per-emotion subdirectories.
        output_dir: Directory where .npy signal files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []  # Will become the CSV

    # Find all video files
    video_exts = {".avi", ".mp4", ".mov", ".mkv", ".flv"}
    all_videos = []
    for emotion_dir in sorted(data_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion_name = emotion_dir.name.lower()
        if emotion_name not in [c.lower() for c in CLASS_NAMES]:
            log.warning(f"Unknown emotion directory: {emotion_dir.name}. Skipping.")
            continue
        label = [c.lower() for c in CLASS_NAMES].index(emotion_name)
        for video_file in sorted(emotion_dir.glob("*")):
            if video_file.suffix.lower() in video_exts:
                all_videos.append((video_file, label, emotion_name))

    log.info(f"Found {len(all_videos)} video clips in {data_dir}")

    # Deterministic train/val split
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(all_videos))
    rng.shuffle(indices)
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_indices = set(indices[:split_idx].tolist())

    success, failed = 0, 0
    for i, (video_path, label, emotion_name) in enumerate(
        tqdm(all_videos, desc="Extracting signals")
    ):
        signals = extract_signals_from_video(video_path)
        if signals is None:
            failed += 1
            continue

        split = "train" if i in train_indices else "val"
        out_name = f"{video_path.stem}_{emotion_name}_label{label}.npy"
        out_path = output_dir / out_name
        np.save(out_path, signals)

        metadata.append(
            {
                "signal_path": str(out_path),
                "video_path": str(video_path),
                "emotion": emotion_name,
                "label": label,
                "split": split,
            }
        )
        success += 1

    # Save metadata CSV
    csv_path = output_dir / "signals_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["signal_path", "video_path", "emotion", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    log.info(f"Done. ✓ {success} succeeded, ✗ {failed} failed.")
    log.info(f"Metadata saved to: {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FAU signals from video dataset.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root dir with per-emotion subfolders",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=SIGNALS_DIR, help="Output dir for .npy files"
    )
    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_dir)
