"""
config.py
─────────
Central configuration file for the Spectrogram FER project.
All paths, hyperparameters, and constants are defined here.
Modify this file to adapt the project to a new dataset or experiment.
"""

from pathlib import Path

# ─── Root Paths ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"
RUNS_DIR = ROOT_DIR / "runs"

# ─── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_DIR = DATA_DIR / "raw"
SIGNALS_DIR = DATA_DIR / "processed" / "signals"
SPECTROGRAMS_DIR = DATA_DIR / "processed" / "spectrograms"
TRAIN_SPEC_DIR = SPECTROGRAMS_DIR / "train"
VAL_SPEC_DIR = SPECTROGRAMS_DIR / "val"

# Ensure critical directories exist
for d in [MODELS_DIR, REPORTS_DIR, RUNS_DIR, SIGNALS_DIR, SPECTROGRAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Emotion Classes ──────────────────────────────────────────────────────────
# RAVDESS Dataset: 8 emotion categories
EMOTION_CLASSES = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fear",
    6: "Disgust",
    7: "Surprise",
}
NUM_CLASSES = len(EMOTION_CLASSES)
CLASS_NAMES = list(EMOTION_CLASSES.values())

# ─── Facial Landmark Signal Configuration ────────────────────────────────────
# MediaPipe Face Mesh landmark indices (478-point model)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png  # noqa: E501

LANDMARK_SIGNALS = {
    # (name, [landmark_idx_A, landmark_idx_B]) → Euclidean distance over time
    "lip_aperture": (13, 14),  # Upper-lower lip center gap (MAR)
    "mouth_width": (61, 291),  # Lip corner distance
    "left_brow_raise": (105, 159),  # Left eyebrow to left eye distance
    "right_brow_raise": (334, 386),  # Right eyebrow to right eye distance
    "left_eye_open": (159, 145),  # Left eye vertical aperture
    "right_eye_open": (386, 374),  # Right eye vertical aperture
    "jaw_open": (152, 10),  # Chin to nose tip (jaw drop proxy)
}

SIGNAL_NAMES = list(LANDMARK_SIGNALS.keys())
NUM_SIGNALS = len(SIGNAL_NAMES)

# Signals mapped to R, G, B channels in the spectrogram image
# Channel 0 (R): mouth-related signals
# Channel 1 (G): brow-related signals
# Channel 2 (B): eye/jaw signals
CHANNEL_SIGNAL_MAP = {
    "R": ["lip_aperture", "mouth_width"],
    "G": ["left_brow_raise", "right_brow_raise"],
    "B": ["left_eye_open", "right_eye_open", "jaw_open"],
}

# ─── Signal Processing (STFT) Configuration ──────────────────────────────────
VIDEO_FPS = 30  # Assumed FPS of input videos
SIGNAL_LENGTH = 90  # Number of frames per clip (3 seconds @ 30fps)
STFT_NPERSEG = 16  # STFT window length (samples)
STFT_NOVERLAP = 8  # STFT overlap (samples)
STFT_NFFT = 32  # FFT size

# ─── Spectrogram Image Configuration ─────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3  # RGB (one channel per signal group)

# ─── Dataset Split ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
RANDOM_SEED = 42

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4  # DataLoader workers (set 0 on Windows if issues)
PIN_MEMORY = True  # Set False if not using CUDA

# ─── Model Configuration ─────────────────────────────────────────────────────
MODEL_NAME = "resnet18"  # Options: "resnet18", "mobilenet_v3_small", "custom_cnn"
PRETRAINED = True  # Use ImageNet pretrained weights
DROPOUT_RATE = 0.5
FC_HIDDEN_DIM = 256

# ─── Checkpoint & Export ──────────────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
ONNX_MODEL_PATH = MODELS_DIR / "best_model.onnx"
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# ─── Augmentation (Training only) ─────────────────────────────────────────────
AUG_HORIZONTAL_FLIP_P = 0.5
AUG_COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
AUG_ROTATION_DEGREES = 5

# ─── Normalization (ImageNet stats for pretrained models) ─────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
