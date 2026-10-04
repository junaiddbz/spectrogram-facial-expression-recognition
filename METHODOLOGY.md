# Methodology: Spectrogram-Based Facial Expression Recognition

---

## Abstract

This document describes the technical methodology underlying a facial expression recognition (FER) system that applies Short-Time Fourier Transform (STFT) spectrogram analysis to facial landmark geometry. The system departs from the dominant frame-classification paradigm by treating facial movement as a set of time-varying scalar signals, converting those signals into 2D frequency-time representations, and classifying the resulting images using a convolutional neural network pretrained on ImageNet. This approach preserves the temporal dynamics of facial expressions, which frame-level methods systematically discard.

The model achieves 91.45% validation accuracy across 8 emotion classes on the RAVDESS Video Speech dataset.

---

## 1. Motivation

Human facial expressions are dynamic events. A genuine smile has a characteristic onset, apex, and offset that unfold over 300–500 milliseconds. Disgust involves a rapid sequence of muscle group activations. These temporal structures carry discriminative information that no single frame can represent.

Speech emotion recognition has demonstrated that converting 1D acoustic signals into 2D time-frequency representations (spectrograms) and classifying them with CNNs is highly effective. The underlying hypothesis of this work is that the same approach applies to facial motion: if facial landmark trajectories are treated as 1D signals, their STFT spectrograms should encode emotion-discriminative frequency and temporal patterns that a CNN can learn to classify.

---

## 2. Dataset

**RAVDESS Video Speech Dataset** (Livingstone & Russo, 2018)

- 24 professional actors (12 male, 12 female)
- 8 emotion classes: Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise
- 2,880 video clips total
- Each clip consists of a single spoken phrase delivered with a target emotion
- 80/20 stratified train/validation split (random seed = 42)

---

## 3. Facial Action Unit Signal Extraction

### 3.1 Landmark Detection

MediaPipe Face Mesh (v0.10.9) is applied to every frame of each video clip. The model returns a set of 478 3D keypoints normalized to the image dimensions. Only the x and y coordinates are used.

### 3.2 Geometric Feature Computation

Seven scalar distances are computed per frame, each corresponding to a Facial Action Unit (FAU) group from the Facial Action Coding System (FACS):

| Signal | Landmark Pair | Muscle Group |
|--------|--------------|--------------|
| `lip_aperture` | 13 - 14 | Orbicularis oris (vertical lip gap) |
| `mouth_width` | 61 - 291 | Zygomaticus major (horizontal lip stretch) |
| `left_brow_raise` | 105 - 159 | Frontalis left (brow elevation) |
| `right_brow_raise` | 334 - 386 | Frontalis right (brow elevation) |
| `left_eye_open` | 159 - 145 | Levator palpebrae left (eyelid aperture) |
| `right_eye_open` | 386 - 374 | Levator palpebrae right (eyelid aperture) |
| `jaw_open` | 152 - 10 | Masseter / digastric (jaw depression) |

### 3.3 Scale Normalization

Raw pixel distances vary with the subject's distance from the camera and image resolution. Each FAU distance at frame t is normalized by the inter-ocular distance at the same frame:

```
d_norm(t) = d_raw(t) / (d_iod(t) + epsilon)
```

where `d_iod(t)` is the Euclidean distance between outer eye corner landmarks 33 and 263, and `epsilon = 1e-6` prevents division by zero. This renders the signals invariant to camera distance and subject head size.

### 3.4 Temporal Resampling

Videos in RAVDESS vary in length and frame rate. To produce fixed-length inputs, each of the 7 normalized signals is resampled to exactly T = 90 samples using linear interpolation. At 30 fps, this corresponds to a 3-second analysis window. The final output per clip is a matrix of shape `[7 x 90]`.

If MediaPipe fails to detect a face in a frame (due to occlusion or extreme head pose), the signal value from the previous valid frame is carried forward.

---

## 4. STFT Spectrogram Generation

### 4.1 Short-Time Fourier Transform

The Short-Time Fourier Transform is applied independently to each of the 7 FAU signals using `scipy.signal.stft` with the following parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window type | Hann | Minimizes spectral leakage |
| Window length (`nperseg`) | 16 samples | Balances time/frequency resolution for short signals |
| Hop size (`noverlap`) | 8 samples | 50% overlap; standard practice |
| FFT size (`nfft`) | 32 | Yields 17 frequency bins per time frame |

The log-magnitude spectrogram is computed as:

```
S[m, k] = 20 * log10(|STFT[m, k]| + epsilon)
```

Each spectrogram is then normalized to the range [0, 255] and stored as uint8.

### 4.2 RGB Channel Fusion

Rather than stacking all 7 spectrograms into a 7-channel tensor (which would require a custom backbone), the 7 signals are averaged into 3 groups and mapped to the R, G, B channels of a standard 3-channel image:

| Channel | Signals Averaged | Anatomical Region |
|---------|-----------------|-------------------|
| Red | `lip_aperture`, `mouth_width` | Mouth |
| Green | `left_brow_raise`, `right_brow_raise` | Brows |
| Blue | `left_eye_open`, `right_eye_open`, `jaw_open` | Eyes and jaw |

This grouping is anatomically motivated: the three channels correspond to the three primary muscular regions that drive visible emotional expression. The fusion produces a single `224 x 224 x 3` PNG image per video clip, directly compatible with standard ImageNet-pretrained CNN architectures.

---

## 5. Model Architecture

### 5.1 Backbone

ResNet-18 (He et al., 2016), pretrained on ImageNet (ILSVRC 2012), is used as the feature extraction backbone. The final average pooling layer produces a 512-dimensional feature vector.

### 5.2 Classification Head

The original 1000-class fully connected layer is replaced with a custom two-layer head:

```
Linear(512, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 8) -> Softmax
```

Dropout with p = 0.5 is applied between the two linear layers as a regularizer.

### 5.3 Two-Stage Training Protocol

Training proceeds in two distinct stages to prevent the large pretrained weights from distorting the randomly initialized classification head in early iterations.

**Stage 1 — Head Warm-up (5 epochs)**

The ResNet-18 backbone is fully frozen. Only the classification head parameters are updated, using AdamW with a learning rate of 1e-3. This allows the head to converge toward a reasonable initialization before the backbone is modified.

**Stage 2 — Full Fine-tuning (up to 50 epochs)**

All layers are unfrozen. AdamW with a base learning rate of 1e-4 and weight decay of 1e-4 is used with a CosineAnnealingLR scheduler (T_max = 50). Training terminates early if validation accuracy does not improve for 10 consecutive epochs. Mixed-precision training (torch.cuda.amp) is used to reduce memory consumption on the GTX 1050 Ti.

### 5.4 Loss Function and Regularization

Cross-entropy loss with label smoothing (epsilon = 0.1) is used throughout. Label smoothing prevents the model from producing overconfident logits on the training set, which improves calibration and generalization on the relatively small RAVDESS dataset.

A WeightedRandomSampler is applied to the training DataLoader to counteract minor class imbalances, ensuring each emotion class is sampled at an equal expected frequency per batch.

---

## 6. Inference and Deployment

After training, the best checkpoint (highest validation accuracy) is exported to ONNX format using `torch.onnx.export` with `opset_version=17`. ONNX Runtime is used for CPU inference in the Streamlit application, removing the PyTorch dependency from the deployment environment.

For real-time webcam inference, a rolling buffer stores the most recent 90 frames of FAU signals. A new prediction is generated every time the buffer fills, producing an updated emotion label and confidence score approximately every 3 seconds at 30 fps.

---

## 7. Evaluation Protocol

All evaluation is performed on the held-out validation split (20% of RAVDESS, stratified by emotion class). The following metrics are reported:

- Per-class precision, recall, and F1-score
- Macro-averaged accuracy and F1
- One-vs-rest ROC curves and AUC scores per class
- t-SNE visualization of the 512-dimensional feature embeddings extracted from the penultimate layer

---

## 8. Limitations and Future Work

**Signal robustness.** MediaPipe Face Mesh can fail on frames with significant motion blur, extreme head rotation (> 45 degrees), or low illumination. The current fallback (carry-forward from the last valid frame) introduces flat segments in the signal that may degrade STFT quality for clips with frequent detection failures.

**Fixed analysis window.** The 90-frame (3-second) window is a fixed design choice. For shorter expressions, this means the majority of the signal is near-zero padding from resampling. Adaptive windowing based on detected expression onset could improve sensitivity.

**Frequency resolution.** With T = 90 samples and nperseg = 16, the frequency resolution of the STFT is coarse. For longer video clips or higher-frequency facial dynamics, increasing the window length and FFT size would produce richer spectrograms.

**Single-face constraint.** The pipeline assumes exactly one face is visible in each frame. Multi-face scenarios are not handled.

---

## References

1. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE*, 13(5), e0196391.
2. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System*. Consulting Psychologists Press.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv:1906.08172*.
5. Allen, J. B. (1977). Short Term Spectral Analysis, Synthesis and Modification by Discrete Fourier Transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 25(3), 235–238.
6. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
7. Muthukumar, P., & Ramakrishnan, A. G. (2020). Speech Emotion Recognition Using Spectrogram Features and Deep Learning. *Procedia Computer Science*.
