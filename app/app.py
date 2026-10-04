"""
app.py
──────
Streamlit Web Application — Spectrogram Facial Expression Recognition
Deployed on Hugging Face Spaces.

Features:
  - Upload a video or use webcam (local only)
  - Displays the generated spectrogram image
  - Shows predicted emotion + confidence bar chart
  - Shows live FAU signal waveforms
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
sys.path.extend([str(ROOT / "src"), str(ROOT / "app")])

from inference import (  # noqa: E402
    EMOTION_EMOJIS,
    FacialSignalBuffer,
    ONNXInferenceEngine,
    predict_from_video,
)

from config import CLASS_NAMES, ONNX_MODEL_PATH, SIGNAL_NAMES  # noqa: E402
from make_spectrograms import signals_to_spectrogram  # noqa: E402

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spectrogram FER | Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f172a; }

  .hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-card .emotion-label {
    font-size: 3rem; margin-bottom: 0.2rem;
  }
  .metric-card .emotion-name {
    font-size: 1.5rem; font-weight: 700; color: #a78bfa;
  }
  .metric-card .confidence {
    font-size: 1rem; color: #64748b; margin-top: 0.3rem;
  }
  .spec-caption {
    font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 0.5rem;
  }
  .pipeline-step {
    background: #1e293b; border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin-bottom: 0.5rem; font-size: 0.9rem; color: #cbd5e1;
  }
  div[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_engine() -> ONNXInferenceEngine | None:
    if not ONNX_MODEL_PATH.exists():
        return None
    return ONNXInferenceEngine(ONNX_MODEL_PATH)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Spectrogram FER")
    st.markdown("---")
    st.markdown("### How It Works")
    for step in [
        "📹 **Input** — Video or webcam frames",
        "🔍 **Landmark Extraction** — MediaPipe Face Mesh (478 pts)",
        "📊 **FAU Signals** — 7 distance signals over time",
        "🌊 **STFT** — Short-Time Fourier Transform → spectrogram",
        "🧠 **CNN** — ResNet-18 classifies the image",
        "🎯 **Emotion** — Predicted with confidence",
    ]:
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "A novel FER system using **signal processing** on facial motion data. "
        "Facial landmark distances over time are converted to spectrograms "
        "and classified by a CNN.\n\n"
        "📂 [GitHub](https://github.com/your-username/spectrogram-fer) | "
        "📄 [Methodology](https://github.com/your-username/spectrogram-fer/blob/main/METHODOLOGY.md)"
    )


# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="hero-title">🎭 Spectrogram FER System</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="hero-sub">Emotion recognition via Signal Processing · '
    "MediaPipe → STFT Spectrograms → ResNet-18</p>",
    unsafe_allow_html=True,
)

engine = load_engine()
if engine is None:
    st.error(
        "⚠️ No trained model found at `models/best_model.onnx`. "
        "Run `python src/train.py` first, then export to ONNX.",
        icon="🚨",
    )
    st.stop()


# ─── Input Tabs ───────────────────────────────────────────────────────────────
tab_upload, tab_webcam, tab_demo = st.tabs(
    ["📁 Upload Video", "📷 Webcam", "🎬 Demo Samples"]
)

result = None

# ── Tab 1: Upload Video ────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a short video clip (3–10 seconds, face clearly visible)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔄 Processing video through pipeline..."):
            result = predict_from_video(tmp_path, engine)


# ── Tab 2: Webcam ─────────────────────────────────────────────────────────────
with tab_webcam:
    st.info(
        "📌 Webcam capture requires running the app **locally** (not supported on Hugging Face Spaces). "  # noqa: E501
        "Run `streamlit run app/app.py` on your machine.",
        icon="ℹ️",
    )
    run_webcam = st.button("▶ Start Webcam Capture (3 seconds)", type="primary")

    if run_webcam:
        cap = cv2.VideoCapture(0)
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
                frame_placeholder.image(
                    frame_rgb, channels="RGB", use_column_width=True
                )
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
                        "emotion_emoji": EMOTION_EMOJIS.get(pred_class, "🎭"),
                        "signal_matrix": signal_matrix,
                        "spectrogram_img": spectrogram_img,
                    }
            else:
                st.warning(
                    "Not enough frames with a detected face. Try again in better lighting."
                )


# ── Tab 3: Demo Samples ────────────────────────────────────────────────────────
with tab_demo:
    st.info(
        "Demo samples can be added to `app/demo_samples/` after training.", icon="📌"
    )


# ─── Results Panel ────────────────────────────────────────────────────────────
if result is not None:
    if "error" in result:
        st.error(result["error"])
    else:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        col_pred, col_spec, col_signals = st.columns([1, 1.4, 1.6])

        # ── Prediction card ───────────────────────────────────────────────
        with col_pred:
            proba = result["proba"]
            pred_class = result["predicted_class"]
            emoji = result["emotion_emoji"]
            confidence = float(np.max(proba)) * 100

            st.markdown(
                f"""
            <div class="metric-card">
              <div class="emotion-label">{emoji}</div>
              <div class="emotion-name">{pred_class}</div>
              <div class="confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Bar chart of all probabilities
            colors = ["#7c3aed" if c == pred_class else "#334155" for c in CLASS_NAMES]
            fig_bar = go.Figure(
                go.Bar(
                    x=CLASS_NAMES,
                    y=(proba * 100).tolist(),
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#cbd5e1", size=11),
                yaxis=dict(
                    title="Probability (%)", range=[0, 105], gridcolor="#334155"
                ),
                xaxis=dict(tickangle=-30),
                margin=dict(l=20, r=20, t=30, b=40),
                height=320,
                title=dict(text="Emotion Probabilities", font=dict(size=13)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Spectrogram image ─────────────────────────────────────────────
        with col_spec:
            st.image(
                result["spectrogram_img"],
                caption="Generated RGB Spectrogram (CNN Input)",
                use_column_width=True,
            )
            st.markdown(
                '<p class="spec-caption">R=Mouth signals · G=Brow signals · B=Eye/Jaw signals</p>',
                unsafe_allow_html=True,
            )

        # ── FAU signal waveforms ──────────────────────────────────────────
        with col_signals:
            st.markdown("**Facial Action Unit Signals**")
            signal_matrix = result["signal_matrix"]
            t = np.linspace(0, signal_matrix.shape[1] / 30, signal_matrix.shape[1])

            palette = [
                "#ef4444",
                "#f97316",
                "#eab308",
                "#22c55e",
                "#06b6d4",
                "#7c3aed",
                "#ec4899",
            ]

            fig_sig = go.Figure()
            for i, name in enumerate(SIGNAL_NAMES):
                fig_sig.add_trace(
                    go.Scatter(
                        x=t.tolist(),
                        y=signal_matrix[i].tolist(),
                        mode="lines",
                        name=name.replace("_", " ").title(),
                        line=dict(color=palette[i % len(palette)], width=1.5),
                    )
                )

            fig_sig.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#cbd5e1", size=10),
                xaxis=dict(title="Time (s)", gridcolor="#334155"),
                yaxis=dict(title="Normalized Distance", gridcolor="#334155"),
                legend=dict(font=dict(size=9), bgcolor="#0f172a"),
                margin=dict(l=20, r=20, t=20, b=40),
                height=360,
            )
            st.plotly_chart(fig_sig, use_container_width=True)
