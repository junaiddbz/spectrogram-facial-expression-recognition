"""
inference.py
────────────
Standalone inference engine for the Streamlit app.

Handles the full pipeline:
  Video/Webcam Frame → MediaPipe Landmarks → FAU Signals → STFT Spectrogram → CNN → Emotion
  Also tracks facial bounding boxes to render an annotated output video.

Uses ONNX Runtime for CPU-only inference (no PyTorch required at runtime).
Uses MediaPipe Face Mesh via the legacy mp.solutions API (mediapipe < 0.10.14).
"""

import sys
import tempfile
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
    """
    Maintains a rolling buffer of FAU distance signals from video frames.
    Also stores facial bounding boxes for rendering output video.
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
        self.bboxes = []  # Stores (x1, y1, x2, y2) for each processed frame
        self.baseline = None  # To store the neutral baseline from the first frames

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
            # Face lost: repeat last values to keep time-series intact
            for name in SIGNAL_NAMES:
                val = self.last_values[name]
                self.buffer[name].append(val)
                if len(self.buffer[name]) > self.window_size:
                    self.buffer[name].pop(0)
            self.bboxes.append(None)
            return False

        lm = results.multi_face_landmarks[0].landmark
        
        # 1. Calculate face bounding box
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        # Add a slight padding to the bounding box
        pad_w, pad_h = int(w * 0.02), int(h * 0.02)
        x1, x2 = max(0, min(xs) - pad_w), min(w, max(xs) + pad_w)
        y1, y2 = max(0, min(ys) - pad_h), min(h, max(ys) + pad_h)
        self.bboxes.append((x1, y1, x2, y2))

        # 2. Extract 3D points
        pts = np.array(
            [[lm_point.x * w, lm_point.y * h, lm_point.z * w] for lm_point in lm]
        )
        inter_ocular = self._euclidean(pts[33], pts[263]) + 1e-6

        # 3. Compute FAUs
        for name, (idx_a, idx_b) in LANDMARK_SIGNALS.items():
            val = self._euclidean(pts[idx_a], pts[idx_b]) / inter_ocular
            self.last_values[name] = val
            self.buffer[name].append(val)
            if len(self.buffer[name]) > self.window_size:
                self.buffer[name].pop(0)
                
        # 4. Set neutral baseline using the first 15 frames to anchor inference
        if self.baseline is None and len(self.buffer[SIGNAL_NAMES[0]]) == 15:
            self.baseline = {n: np.mean(self.buffer[n][:15]) for n in SIGNAL_NAMES}
            
        return True

    def is_ready(self) -> bool:
        # Ready as soon as we have a neutral baseline (first 15 frames / 0.5 seconds)
        # This completely eliminates the 3-second startup "Buffering..." lag.
        return self.baseline is not None

    def get_signal_matrix(self) -> np.ndarray:
        """Return (NUM_SIGNALS, SIGNAL_LENGTH) signal matrix from current buffer."""
        matrix = []
        for name in SIGNAL_NAMES:
            if self.baseline is not None:
                # We use ONLY the last 30 frames (1 second) of actual facial history.
                # This guarantees that old emotions don't linger in the 3-second window
                # and cause a massive prediction lag.
                actual_history = np.array(self.buffer[name][-30:], dtype=np.float32)
                hist_len = len(actual_history)
                
                # Fill the window with the Neutral baseline
                base = self.baseline[name]
                raw = np.full(SIGNAL_LENGTH, base, dtype=np.float32)
                
                if hist_len > 0:
                    # Place actual recent history at the very end of the 90-frame window
                    raw[-hist_len:] = actual_history
                    
                    # Create a smooth 10-frame transition from Neutral to the start of this history
                    ramp_start = SIGNAL_LENGTH - hist_len - 10
                    if ramp_start >= 0:
                        ramp = np.linspace(base, actual_history[0], 10)
                        raw[ramp_start : ramp_start + 10] = ramp
                
                # No resampling needed since `raw` is exactly perfectly crafted to SIGNAL_LENGTH
                matrix.append(raw)
            else:
                # Fallback if somehow called before baseline
                raw = np.array(self.buffer[name][-self.window_size :], dtype=np.float32)
                if len(raw) == 0:
                    raw = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
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
    Uses a sliding window to predict changing emotions across the entire video dynamically.
    """
    buffer = FacialSignalBuffer()
    cap = cv2.VideoCapture(video_path)

    frame_predictions = []
    last_pred = ("Buffering...", 0.0)
    frame_count = 0
    
    # Temporal smoothing to prevent jittery predictions
    proba_history = []  
    SMOOTHING_WINDOW = 7  # Average last 7 predictions (~0.7 seconds of video)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        buffer.push_frame(frame)
        
        # Sliding window dynamic prediction: predict every 3 frames to optimize speed
        if buffer.is_ready() and frame_count % 3 == 0:
            signal_matrix = buffer.get_signal_matrix()
            spectrogram_img = signals_to_spectrogram(signal_matrix)
            proba = engine.predict(spectrogram_img)
            
            proba_history.append(proba)
            if len(proba_history) > SMOOTHING_WINDOW:
                proba_history.pop(0)
                
            smoothed_proba = np.mean(proba_history, axis=0)
            pred_idx = int(np.argmax(smoothed_proba))
            last_pred = (CLASS_NAMES[pred_idx], float(np.max(smoothed_proba)) * 100)
            
        frame_predictions.append(last_pred)
        frame_count += 1

    cap.release()
    buffer.close()

    if not buffer.is_ready():
        return {
            "error": "Could not extract enough facial landmarks from the video. "
            "Ensure the face is clearly visible."
        }

    # 1. Final state for the static UI displays (last 3 seconds)
    signal_matrix = buffer.get_signal_matrix()
    spectrogram_img = signals_to_spectrogram(signal_matrix)
    proba = engine.predict(spectrogram_img)
    pred_idx = int(np.argmax(proba))
    final_class = CLASS_NAMES[pred_idx]

    # 2. Render Annotated Output Video with dynamic labels
    out_path = tempfile.NamedTemporaryFile(suffix=".webm", delete=False).name
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    box_color = (241, 102, 99) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(buffer.bboxes) and buffer.bboxes[frame_idx] is not None:
            x1, y1, x2, y2 = buffer.bboxes[frame_idx]
            
            # Get the dynamic prediction for THIS exact frame
            dyn_class, dyn_conf = frame_predictions[frame_idx]
            
            # Draw Face Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw Emotion Text Background
            if dyn_class == "Buffering...":
                text = dyn_class
            else:
                text = f"{dyn_class} {dyn_conf:.1f}%"
                
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            bg_y1 = max(0, y1 - th - 10)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, bg_y1 + th + 10), box_color, -1)
            
            cv2.putText(frame, text, (x1 + 5, bg_y1 + th + 5), font, font_scale, (255, 255, 255), thickness)
            
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return {
        "proba": proba,
        "predicted_class": final_class,
        "signal_matrix": signal_matrix,
        "spectrogram_img": spectrogram_img,
        "annotated_video_path": out_path,
    }
