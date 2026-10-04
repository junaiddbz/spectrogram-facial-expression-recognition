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
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / "src"), str(APP_DIR)])

from inference import (  # noqa: E402
    FacialSignalBuffer,
    ONNXInferenceEngine,
    predict_from_video,
)
from config import CLASS_NAMES, ONNX_MODEL_PATH, SIGNAL_NAMES  # noqa: E402
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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp { background: #0a0f1e; }
  section[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid #1a2740;
  }

  /* ── Hero ── */
  .hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(120deg, #6366f1 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.35rem;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    letter-spacing: 0.02em;
    margin-bottom: 0;
  }

  /* ── Prediction card ── */
  .pred-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
  }
  .pred-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.8rem;
  }
  .pred-emotion {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
  }
  .pred-confidence {
    font-size: 0.85rem;
    color: #6366f1;
    font-weight: 500;
  }
  .confidence-bar-bg {
    background: #1e293b;
    border-radius: 100px;
    height: 6px;
    margin: 0.8rem 0 0;
    overflow: hidden;
  }
  .confidence-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    transition: width 0.4s ease;
  }

  /* ── Pipeline steps ── */
  .pipeline-step {
    background: #111827;
    border-left: 2px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
  }
  .pipeline-step b { color: #cbd5e1; }

  /* ── Section label ── */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
  }

  /* ── Divider ── */
  hr { border-color: #1a2740 !important; }

  /* ── Demo card ── */
  .demo-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.2s;
  }
  .demo-card:hover { border-color: #6366f1; transform: translateY(-2px); }
  .demo-card .dc-emotion { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
  .demo-card .dc-desc { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; }

  /* ── Tabs ── */
  button[data-baseweb="tab"] { font-size: 0.85rem !important; }
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
st.markdown('<h1 class="hero-title">Spectrogram FER System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Emotion recognition via Signal Processing &nbsp;·&nbsp; '
    "MediaPipe &rarr; STFT Spectrograms &rarr; ResNet-18</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

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
            CAPTURE_FRAMES = 90

            for i in range(CAPTURE_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
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
                    result = {
                        "proba": proba,
                        "predicted_class": pred_class,
                        "signal_matrix": signal_matrix,
                        "spectrogram_img": spectrogram_img,
                    }
            else:
                st.warning(
                    "Not enough frames with a detected face. "
                    "Try again in better lighting."
                )


# ── Tab 3: Demo Samples ────────────────────────────────────────────────────────
DEMO_META = {
    "Happy": "Wide smile, raised lip corners",
    "Surprised": "Raised brows, open jaw, wide eyes",
    "Neutral": "Relaxed face, baseline signals",
    "Angry": "Furrowed brows, tense jaw",
}

with tab_demo:
    if not DEMO_DIR.exists() or not list(DEMO_DIR.glob("*.npy")):
        st.warning(
            "Demo samples not found. "
            "Run `python app/generate_demo_samples.py` to generate them."
        )
    else:
        st.markdown(
            '<p class="section-label">Select a pre-generated signal pattern to run inference</p>',
            unsafe_allow_html=True,
        )
        demo_files = {p.stem.capitalize(): p for p in sorted(DEMO_DIR.glob("*.npy"))}
        cols = st.columns(len(demo_files))
        selected_demo = None

        for col, (name, path) in zip(cols, demo_files.items()):
            with col:
                desc = DEMO_META.get(name, "Synthetic signal pattern")
                st.markdown(
                    f'<div class="demo-card">'
                    f'<div class="dc-emotion">{name}</div>'
                    f'<div class="dc-desc">{desc}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                if st.button(f"Run — {name}", key=f"demo_{name}", use_container_width=True):
                    selected_demo = path

        if selected_demo is not None:
            with st.spinner("Running inference on demo sample..."):
                signal_matrix = np.load(selected_demo)
                spectrogram_img = signals_to_spectrogram(signal_matrix)
                proba = engine.predict(spectrogram_img)
                pred_idx = int(np.argmax(proba))
                pred_class = CLASS_NAMES[pred_idx]
                result = {
                    "proba": proba,
                    "predicted_class": pred_class,
                    "signal_matrix": signal_matrix,
                    "spectrogram_img": spectrogram_img,
                }


# ─── Results Panel ────────────────────────────────────────────────────────────
if result is not None:
    if "error" in result:
        st.error(result["error"])
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Analysis Results</p>', unsafe_allow_html=True)

        if "annotated_video_path" in result:
            st.video(result["annotated_video_path"])
            st.markdown("<br>", unsafe_allow_html=True)

        col_pred, col_spec, col_signals = st.columns([1, 1.3, 1.7])

        # ── Prediction card ───────────────────────────────────────────────
        with col_pred:
            proba = result["proba"]
            pred_class = result["predicted_class"]
            confidence = float(np.max(proba)) * 100

            st.markdown(
                f"""
            <div class="pred-card">
              <div class="pred-label">Predicted Emotion</div>
              <div class="pred-emotion">{pred_class}</div>
              <div class="pred-confidence">{confidence:.1f}% confidence</div>
              <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{confidence:.1f}%"></div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            colors = ["#6366f1" if c == pred_class else "#1e293b" for c in CLASS_NAMES]
            fig_bar = go.Figure(
                go.Bar(
                    x=CLASS_NAMES,
                    y=(proba * 100).tolist(),
                    marker_color=colors,
                    text=[f"{p*100:.0f}%" for p in proba],
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                )
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1526",
                font=dict(color="#94a3b8", size=10, family="Inter"),
                yaxis=dict(
                    title="Confidence (%)",
                    range=[0, 110],
                    gridcolor="#1a2740",
                    zeroline=False,
                    tickfont=dict(size=9),
                ),
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                margin=dict(l=10, r=10, t=10, b=40),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Spectrogram image ─────────────────────────────────────────────
        with col_spec:
            st.markdown(
                '<p class="section-label">Generated Spectrogram</p>', unsafe_allow_html=True
            )
            st.image(
                result["spectrogram_img"],
                caption="RGB Spectrogram — CNN Input  |  R=Mouth  G=Brow  B=Eye/Jaw",
                use_container_width=True,
            )

        # ── FAU signal waveforms ──────────────────────────────────────────
        with col_signals:
            st.markdown(
                '<p class="section-label">Facial Action Unit Signals</p>',
                unsafe_allow_html=True,
            )
            signal_matrix = result["signal_matrix"]
            t = np.linspace(0, signal_matrix.shape[1] / 30, signal_matrix.shape[1])

            palette = [
                "#6366f1",
                "#38bdf8",
                "#34d399",
                "#f59e0b",
                "#f87171",
                "#a78bfa",
                "#fb923c",
            ]

            fig_sig = go.Figure()
            for i, name in enumerate(SIGNAL_NAMES):
                fig_sig.add_trace(
                    go.Scatter(
                        x=t.tolist(),
                        y=signal_matrix[i].tolist(),
                        mode="lines",
                        name=name.replace("_", " ").title(),
                        line=dict(color=palette[i % len(palette)], width=1.8),
                    )
                )

            fig_sig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1526",
                font=dict(color="#94a3b8", size=10, family="Inter"),
                xaxis=dict(title="Time (s)", gridcolor="#1a2740", zeroline=False),
                yaxis=dict(
                    title="Normalised Distance",
                    gridcolor="#1a2740",
                    zeroline=False,
                ),
                legend=dict(
                    font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#1a2740",
                    borderwidth=1,
                ),
                margin=dict(l=10, r=10, t=10, b=40),
                height=340,
            )
            st.plotly_chart(fig_sig, use_container_width=True)
