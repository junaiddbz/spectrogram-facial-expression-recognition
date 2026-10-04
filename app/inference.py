"""
inference.py
────────────
Standalone inference engine for the Streamlit app.

Handles the full pipeline:
  Video/Webcam Frame → MediaPipe Landmarks → FAU Signals → STFT Spectrogram → CNN → Emotion

Supports both PyTorch (.pth) and ONNX (.onnx) backends.
ONNX is preferred for CPU-only environments (Hugging Face Spaces).
"""

import sys
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

mp_face_mesh = mp.solutions.face_mesh

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Contempt": "😒",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sadness": "😢",
    "Surprise": "😲",
}


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
        """
        Args:
            spectrogram_image: RGB uint8 array (H, W, 3)
        Returns:
            proba: Softmax probabilities (NUM_CLASSES,)
        """
        tensor = self._preprocess(spectrogram_image)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        proba = self._softmax(logits[0])
        return proba

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img).resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - np.array(NORMALIZE_MEAN)) / np.array(NORMALIZE_STD)
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return arr[np.newaxis, :].astype(np.float32)  # add batch dim

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()


# ─── Frame Buffer & Signal Extraction ────────────────────────────────────────
class FacialSignalBuffer:
    """
    Maintains a rolling buffer of FAU distance signals from video frames.
    Used for real-time webcam inference.
    """

    def __init__(self, window_size: int = SIGNAL_LENGTH) -> None:
        self.window_size = window_size
        self.buffer = {name: [] for name in SIGNAL_NAMES}
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_values = {name: 0.0 for name in SIGNAL_NAMES}

    def _euclidean(self, p1, p2) -> float:
        return float(np.linalg.norm(p1 - p2))

    def push_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Process one BGR frame. Returns True if face was detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            for name in SIGNAL_NAMES:
                val = self.last_values[name]
                self.buffer[name].append(val)
                if len(self.buffer[name]) > self.window_size:
                    self.buffer[name].pop(0)
            return False

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[lm_point.x * w, lm_point.y * h, lm_point.z * w] for lm_point in lm])
        inter_ocular = self._euclidean(pts[33], pts[263]) + 1e-6

        for name, (idx_a, idx_b) in LANDMARK_SIGNALS.items():
            val = self._euclidean(pts[idx_a], pts[idx_b]) / inter_ocular
            self.last_values[name] = val
            self.buffer[name].append(val)
            if len(self.buffer[name]) > self.window_size:
                self.buffer[name].pop(0)
        return True

    def is_ready(self) -> bool:
        """Buffer has enough frames for a full spectrogram."""
        return all(len(v) >= self.window_size for v in self.buffer.values())

    def get_signal_matrix(self) -> np.ndarray:
        """Return (NUM_SIGNALS, SIGNAL_LENGTH) signal matrix from current buffer."""
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
        self.face_mesh.close()


# ─── Full Inference Pipeline ──────────────────────────────────────────────────
def predict_from_video(video_path: str, engine: ONNXInferenceEngine) -> dict:
    """
    Run the full pipeline on an uploaded video file.

    Returns:
        dict with keys: proba, predicted_class, emotion_emoji, signal_matrix, spectrogram_img
    """
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
            "Ensure the face is clearly visible."
        }

    signal_matrix = buffer.get_signal_matrix()
    spectrogram_img = signals_to_spectrogram(signal_matrix)
    proba = engine.predict(spectrogram_img)
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "proba": proba,
        "predicted_class": pred_class,
        "emotion_emoji": EMOTION_EMOJIS.get(pred_class, "🎭"),
        "signal_matrix": signal_matrix,
        "spectrogram_img": spectrogram_img,
    }
