"""
app.py
──────
Streamlit Web Application — Spectrogram Facial Expression Recognition
Deployed on Streamlit Community Cloud.

Features:
  - Upload a video or use webcam (local only)
  - Displays the generated spectrogram image
  - Shows predicted emotion + confidence bar chart
  - Shows live FAU signal waveforms
  - Pre-loaded demo samples
"""

import sys
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import streamlit as st

def array_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / "src"), str(APP_DIR)])

from inference import (  # noqa: E402
    FacialSignalBuffer,
    ONNXInferenceEngine,
    predict_from_video,
)
from config import CLASS_NAMES, ONNX_MODEL_PATH  # noqa: E402
from make_spectrograms import signals_to_spectrogram  # noqa: E402

GITHUB_URL = "https://github.com/junaiddbz/spectrogram-facial-expression-recognition"
DEMO_DIR = APP_DIR / "demo_samples"

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spectrogram FER",
    page_icon="assets/favicon.png" if (APP_DIR / "assets/favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
  }

  /* ── Background ── */
  .stApp { 
      background: radial-gradient(circle at top, #1e1b4b, #020617);
  }
  section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
  }

  /* ── Hero ── */
  .hero-container {
      text-align: center;
      padding: 3rem 0 2.5rem;
  }
  .hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 4px 12px rgba(168, 85, 247, 0.2));
    line-height: 1.1;
  }
  .hero-sub {
    font-size: 1.15rem;
    color: #94a3b8;
    letter-spacing: 0.02em;
    font-weight: 400;
  }

  /* ── Custom Result Card ── */
  .result-card {
      background: rgba(30, 41, 59, 0.4);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      overflow: hidden;
      margin-bottom: 24px;
      transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
  }
  .result-card:hover {
      transform: translateY(-6px) scale(1.02);
      border-color: rgba(255, 255, 255, 0.2);
      box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.7), 0 0 20px rgba(168, 85, 247, 0.15);
  }
  .rc-image {
      width: 100%;
      padding-top: 100%; /* 1:1 Aspect Ratio */
      background-size: cover;
      background-position: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }
  .rc-content {
      padding: 20px;
      text-align: center;
  }
  .rc-timestamp {
      font-size: 0.75rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 600;
      margin-bottom: 8px;
  }
  .rc-emotion {
      font-size: 1.8rem;
      font-weight: 800;
      margin: 0 0 14px 0;
      letter-spacing: 0.5px;
      text-shadow: 0 2px 10px rgba(0,0,0,0.4);
  }
  .rc-confidence {
      height: 6px;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 8px;
  }
  .rc-conf-fill {
      height: 100%;
      border-radius: 10px;
      transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .rc-conf-text {
      font-size: 0.85rem;
      color: #cbd5e1;
      font-weight: 500;
  }

  /* ── Pipeline steps ── */
  .pipeline-step {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(10px);
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
    border-top: 1px solid rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.02);
  }
  .pipeline-step b { color: #f8fafc; font-weight: 600; }

  /* ── Section label ── */
  .section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
  }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.05) !important; }

  /* ── Tabs Customization ── */
  [data-baseweb="tab-list"] {
      gap: 10px;
      background: rgba(15, 23, 42, 0.6);
      padding: 6px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      margin-bottom: 1.5rem;
  }
  [data-baseweb="tab"] {
      background: transparent !important;
      border-radius: 8px !important;
      padding: 10px 24px !important;
      font-size: 0.95rem !important;
      font-weight: 600 !important;
      color: #94a3b8 !important;
      border: none !important;
  }
  [aria-selected="true"] {
      background: rgba(255, 255, 255, 0.1) !important;
      color: #fff !important;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  }
  [data-baseweb="tab-highlight"] {
      display: none;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_engine() -> ONNXInferenceEngine | None:
    if not ONNX_MODEL_PATH.exists():
        return None
    return ONNXInferenceEngine(ONNX_MODEL_PATH)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-bottom:0.1rem;">'
        "Spectrogram FER</p>"
        '<p style="font-size:0.75rem;color:#475569;margin-top:0;">Facial Expression Recognition</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<p class="section-label">How It Works</p>', unsafe_allow_html=True)
    steps = [
        ("<b>Input</b>", "Video or webcam frames"),
        ("<b>Landmark Extraction</b>", "MediaPipe Face Mesh (478 pts)"),
        ("<b>FAU Signals</b>", "7 distance signals over time"),
        ("<b>STFT</b>", "Short-Time Fourier Transform"),
        ("<b>CNN</b>", "ResNet-18 classifies the spectrogram"),
        ("<b>Prediction</b>", "Emotion + confidence score"),
    ]
    for title, desc in steps:
        st.markdown(
            f'<div class="pipeline-step">{title}<br><span style="color:#64748b">{desc}</span></div>',
            unsafe_allow_html=True,
        )


    st.divider()
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown(
        "A novel approach to Facial Expression Recognition using **signal processing** "
        "on facial motion data. Landmark distances over time are converted to spectrograms "
        "and classified by a CNN.",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<a href="{GITHUB_URL}" target="_blank" style="color:#6366f1;font-size:0.85rem;'
        'text-decoration:none;font-weight:500;">View on GitHub &rarr;</a>',
        unsafe_allow_html=True,
    )


# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Spectrogram FER System</div>
    <div class="hero-sub">Emotion recognition via Signal Processing &nbsp;·&nbsp; MediaPipe &rarr; STFT Spectrograms &rarr; ResNet-18</div>
</div>
""", unsafe_allow_html=True)

engine = load_engine()
if engine is None:
    st.error(
        "No trained model found at `models/best_model.onnx`. "
        "Run `python src/train.py` first, then export with `python src/export_onnx.py`."
    )
    st.stop()


# ─── Input Tabs ───────────────────────────────────────────────────────────────
tab_upload, tab_webcam, tab_demo = st.tabs(["Upload Video", "Webcam", "Demo Samples"])

result = None

# ── Tab 1: Upload Video ────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a short video clip (3–10 seconds, face clearly visible)",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Processing video through pipeline..."):
            result = predict_from_video(tmp_path, engine)


# ── Tab 2: Webcam ─────────────────────────────────────────────────────────────
with tab_webcam:
    st.info(
        "Webcam capture requires running the app **locally**. "
        "Run `streamlit run app/app.py` on your machine.",
    )
    run_webcam = st.button("Start Webcam Capture (3 seconds)", type="primary")

    if run_webcam:
        # Try default backend first, then DirectShow for Windows
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            buffer = FacialSignalBuffer()
            frame_placeholder = st.empty()
            progress = st.progress(0, text="Capturing frames...")
            CAPTURE_FRAMES = 100

            frames_bgr = []
            for i in range(CAPTURE_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
                buffer.push_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                progress.progress(
                    (i + 1) / CAPTURE_FRAMES,
                    text=f"Capturing... {i+1}/{CAPTURE_FRAMES} frames",
                )

            cap.release()
            buffer.close()
            frame_placeholder.empty()

            if buffer.is_ready():
                with st.spinner("Running inference..."):
                    signal_matrix = buffer.get_signal_matrix()
                    spectrogram_img = signals_to_spectrogram(signal_matrix)
                    proba = engine.predict(spectrogram_img)
                    pred_idx = int(np.argmax(proba))
                    pred_class = CLASS_NAMES[pred_idx]
                    
                    rgb_img = cv2.cvtColor(frames_bgr[45], cv2.COLOR_BGR2RGB) if len(frames_bgr) > 45 else frame_rgb
                    
                    result = [{
                        "timestamp": "Live Capture",
                        "image": rgb_img,
                        "emotion": pred_class,
                        "confidence": float(np.max(proba)) * 100
                    }]
            else:
                st.warning(
                    "Not enough frames with a detected face. " "Try again in better lighting."
                )


# ── Tab 3: Demo Samples ────────────────────────────────────────────────────────
with tab_demo:
    demo_videos = sorted([p for p in DEMO_DIR.glob("*.mp4")] + [p for p in DEMO_DIR.glob("*.avi")])
    if not DEMO_DIR.exists() or not demo_videos:
        st.warning("No demo videos found in `app/demo_samples/`.")
    else:
        st.markdown(
            '<p class="section-label">Select a demo video to run inference</p>',
            unsafe_allow_html=True,
        )
        
        # Group by emotion
        emotions = sorted(list(set([p.stem.split('_')[0].capitalize() for p in demo_videos])))
        selected_emotion = st.selectbox("Select Emotion Category", emotions)
        
        # Get videos for selected emotion
        filtered_videos = [p for p in demo_videos if p.stem.split('_')[0].capitalize() == selected_emotion]
        
        if not filtered_videos:
            st.info("No videos found for this emotion.")
        else:
            cols = st.columns(len(filtered_videos))
            selected_demo_video = None
            for col, vid_path in zip(cols, filtered_videos):
                with col:
                    st.video(str(vid_path))
                    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                    if st.button(f"Run {vid_path.stem}", key=f"demo_{vid_path.stem}", use_container_width=True):
                        selected_demo_video = str(vid_path)
                        
            if selected_demo_video is not None:
                with st.spinner(f"Running inference on {Path(selected_demo_video).name}..."):
                    result = predict_from_video(selected_demo_video, engine)


# ─── Results Panel ────────────────────────────────────────────────────────────
if result is not None:
    if isinstance(result, dict) and "error" in result:
        st.error(result["error"])
    elif isinstance(result, list):
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label" style="text-align:center; color:#94a3b8;">Temporal Analysis Report</p>', unsafe_allow_html=True)
        
        # Display nicely in a grid
        cols_per_row = 5
        for i in range(0, len(result), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(result):
                    item = result[i + j]
                    with col:
                        # Vibrant colors for dark mode glass cards
                        emotion = item['emotion']
                        color = "#34d399" if emotion in ["Happy", "Surprise"] else "#fb7185" if emotion in ["Angry", "Disgust", "Fear"] else "#60a5fa" if emotion == "Sad" else "#a78bfa"
                        
                        img_b64 = array_to_base64(item["image"])
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="rc-image" style="background-image: url('data:image/jpeg;base64,{img_b64}')"></div>
                            <div class="rc-content">
                                <div class="rc-timestamp">⏱ {item['timestamp']}</div>
                                <h3 class="rc-emotion" style="color: {color};">{emotion}</h3>
                                <div class="rc-confidence">
                                    <div class="rc-conf-fill" style="width: {item['confidence']}%; background: {color};"></div>
                                </div>
                                <div class="rc-conf-text">{item['confidence']:.1f}% Confidence</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
