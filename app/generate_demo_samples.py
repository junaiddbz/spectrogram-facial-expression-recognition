"""
Generate synthetic demo signal samples for the Streamlit app.
Run once: python app/generate_demo_samples.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEMO_DIR = Path(__file__).resolve().parent / "demo_samples"
DEMO_DIR.mkdir(exist_ok=True)

SIGNAL_LENGTH = 90
NUM_SIGNALS = 7
t = np.linspace(0, 3, SIGNAL_LENGTH)


def smooth(sig, window=7):
    return np.convolve(sig, np.ones(window) / window, mode="same")


def make_baseline():
    return np.clip(np.random.normal(0.3, 0.02, SIGNAL_LENGTH), 0.1, 0.6)


DEMOS = {
    "Happy": {
        # lip_aperture: rises (open mouth smile)
        # mouth_width: high (wide smile)
        # left/right_brow_raise: slightly raised
        # left/right_eye_open: normal
        # jaw_open: moderate
        0: smooth(np.clip(0.35 + 0.25 * np.sin(np.pi * t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.1, 0.9)),
        1: smooth(np.clip(0.55 + 0.1 * np.sin(np.pi * t / 3) + np.random.normal(0, 0.015, SIGNAL_LENGTH), 0.3, 0.9)),
        2: smooth(make_baseline() + 0.08),
        3: smooth(make_baseline() + 0.08),
        4: smooth(make_baseline()),
        5: smooth(make_baseline()),
        6: smooth(np.clip(0.25 + 0.15 * np.sin(np.pi * t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.1, 0.6)),
    },
    "Surprised": {
        # jaw_open: high
        # eye_open: high
        # brow_raise: high
        0: smooth(np.clip(0.5 + 0.3 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.95)),
        1: smooth(np.clip(0.45 + 0.2 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.85)),
        2: smooth(np.clip(0.5 + 0.25 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.95)),
        3: smooth(np.clip(0.5 + 0.25 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.95)),
        4: smooth(np.clip(0.4 + 0.25 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.9)),
        5: smooth(np.clip(0.4 + 0.25 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.9)),
        6: smooth(np.clip(0.45 + 0.35 * (t / 3) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.2, 0.95)),
    },
    "Neutral": {
        0: smooth(make_baseline()),
        1: smooth(make_baseline() + 0.05),
        2: smooth(make_baseline()),
        3: smooth(make_baseline()),
        4: smooth(make_baseline()),
        5: smooth(make_baseline()),
        6: smooth(make_baseline()),
    },
    "Angry": {
        # brow_raise: low (furrowed), lip_aperture: tense slight open
        # eye_open: narrowed slightly
        0: smooth(np.clip(0.15 + 0.1 * np.abs(np.sin(2 * np.pi * t)) + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.05, 0.5)),
        1: smooth(np.clip(0.3 + 0.05 * np.sin(np.pi * t) + np.random.normal(0, 0.015, SIGNAL_LENGTH), 0.1, 0.5)),
        2: smooth(np.clip(0.15 + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.05, 0.35)),
        3: smooth(np.clip(0.15 + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.05, 0.35)),
        4: smooth(np.clip(0.2 + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.1, 0.4)),
        5: smooth(np.clip(0.2 + np.random.normal(0, 0.02, SIGNAL_LENGTH), 0.1, 0.4)),
        6: smooth(np.clip(0.12 + np.random.normal(0, 0.015, SIGNAL_LENGTH), 0.05, 0.3)),
    },
}

np.random.seed(42)
for emotion, signals in DEMOS.items():
    matrix = np.stack([signals[i] for i in range(NUM_SIGNALS)], axis=0).astype(np.float32)
    path = DEMO_DIR / f"{emotion.lower()}.npy"
    np.save(path, matrix)
    print(f"Saved {path}  shape={matrix.shape}")

print("Done.")
