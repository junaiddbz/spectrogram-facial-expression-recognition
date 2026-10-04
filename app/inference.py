"""
inference.py
────────────
Standalone inference engine for the Streamlit app.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.30) for face landmark detection.
Uses ONNX Runtime for CPU-only inference (no PyTorch required at runtime).
"""

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import (  # noqa: E402
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    LANDMARK_SIGNALS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ONNX_MODEL_PATH,
    SIGNAL_LENGTH,
    SIGNAL_NAMES,
)
from make_spectrograms import signals_to_spectrogram  # noqa: E402

# ─── MediaPipe Task Model ─────────────────────────────────────────────────────
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_TASK_MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


def _ensure_task_model() -> Path:
    if not _TASK_MODEL_PATH.exists():
        urllib.request.urlretrieve(_TASK_MODEL_URL, _TASK_MODEL_PATH)
    return _TASK_MODEL_PATH


# ─── ONNX Inference Engine ────────────────────────────────────────────────────
class ONNXInferenceEngine:
    """Fast CPU inference using ONNX Runtime."""

    def __init__(self, model_path: Path = ONNX_MODEL_PATH) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, spectrogram_image: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(spectrogram_image)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        return self._softmax(logits[0])

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img).resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - np.array(NORMALIZE_MEAN)) / np.array(NORMALIZE_STD)
        arr = arr.transpose(2, 0, 1)
        return arr[np.newaxis, :].astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()


# ─── Frame Buffer & Signal Extraction ────────────────────────────────────────
class FacialSignalBuffer:
    """Rolling buffer of FAU signals. Uses MediaPipe Tasks FaceLandmarker API."""

    def __init__(self, window_size: int = SIGNAL_LENGTH) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.window_size = window_size
        self.buffer = {name: [] for name in SIGNAL_NAMES}
        self.last_values = {name: 0.0 for name in SIGNAL_NAMES}

        task_path = _ensure_task_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_idx = 0

    def _euclidean(self, p1, p2) -> float:
        return float(np.linalg.norm(p1 - p2))

    def push_frame(self, frame_bgr: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self._frame_idx * (1000 / 30))
        self._frame_idx += 1
        results = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            for name in SIGNAL_NAMES:
                self.buffer[name].append(self.last_values[name])
                if len(self.buffer[name]) > self.window_size:
                    self.buffer[name].pop(0)
            return False

        lm = results.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm])
        inter_ocular = self._euclidean(pts[33], pts[263]) + 1e-6

        for name, (idx_a, idx_b) in LANDMARK_SIGNALS.items():
            val = self._euclidean(pts[idx_a], pts[idx_b]) / inter_ocular
            self.last_values[name] = val
            self.buffer[name].append(val)
            if len(self.buffer[name]) > self.window_size:
                self.buffer[name].pop(0)
        return True

    def is_ready(self) -> bool:
        return all(len(v) >= self.window_size for v in self.buffer.values())

    def get_signal_matrix(self) -> np.ndarray:
        matrix = []
        for name in SIGNAL_NAMES:
            raw = np.array(self.buffer[name][-self.window_size :], dtype=np.float32)
            resampled = np.interp(
                np.linspace(0, len(raw) - 1, SIGNAL_LENGTH),
                np.arange(len(raw)),
                raw,
            )
            matrix.append(resampled)
        return np.array(matrix, dtype=np.float32)

    def close(self) -> None:
        self._detector.close()


# ─── Full Inference Pipeline ──────────────────────────────────────────────────
def predict_from_video(video_path: str, engine: ONNXInferenceEngine) -> dict:
    buffer = FacialSignalBuffer()
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.push_frame(frame)

    cap.release()
    buffer.close()

    if not buffer.is_ready():
        return {
            "error": "Could not extract enough facial landmarks from the video. "
            "Ensure the face is clearly visible and at least 3 seconds long."
        }

    signal_matrix = buffer.get_signal_matrix()
    spectrogram_img = signals_to_spectrogram(signal_matrix)
    proba = engine.predict(spectrogram_img)
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "proba": proba,
        "predicted_class": pred_class,
        "signal_matrix": signal_matrix,
        "spectrogram_img": spectrogram_img,
    }
