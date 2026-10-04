"""
evaluate.py
───────────
Evaluation and visualization utilities.

Generates:
  - Confusion matrix heatmap
  - Per-class classification report (precision, recall, F1)
  - ROC curves (one-vs-rest, per class)
  - t-SNE visualization of learned feature embeddings
  - Training curve plots from TensorBoard logs

Usage:
    python src/evaluate.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    MODEL_NAME,
    NUM_CLASSES,
    REPORTS_DIR,
)
from dataset import SpectrogramDataset, VAL_SPEC_DIR, get_val_transform
from model import build_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Shared plot style ────────────────────────────────────────────────────────
DARK_BG   = "#0f172a"
PANEL_BG  = "#1e293b"
TEXT_CLR  = "#f1f5f9"
ACCENT    = "#7c3aed"
GRID_CLR  = "#334155"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "text.color":       TEXT_CLR,
    "grid.color":       GRID_CLR,
    "font.family":      "DejaVu Sans",
})


# ─── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader and collect predictions.

    Returns:
        y_true:  Ground truth labels (N,)
        y_pred:  Predicted class labels (N,)
        y_proba: Softmax probabilities (N, NUM_CLASSES)
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_proba.extend(proba)

    return np.array(all_true), np.array(all_pred), np.array(all_proba)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate layer embeddings for t-SNE visualization."""
    model.eval()
    embeddings, labels_list = [], []
    hooks = []

    # Register a forward hook on the Global Average Pooling output
    activation = {}
    def _hook(module, inp, out):
        activation["embed"] = out.squeeze().detach().cpu().numpy()

    # Find the AdaptiveAvgPool or equivalent
    for name, module in model.named_modules():
        if isinstance(module, (nn.AdaptiveAvgPool2d,)):
            hooks.append(module.register_forward_hook(_hook))
            break

    for images, lbls in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device, non_blocking=True)
        model(images)
        if "embed" in activation:
            embed = activation["embed"]
            if embed.ndim == 1:
                embed = embed[np.newaxis, :]
            embeddings.append(embed)
        labels_list.extend(lbls.numpy())

    for h in hooks:
        h.remove()

    return np.vstack(embeddings), np.array(labels_list)


# ─── Plot Functions ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor=GRID_CLR,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Spectrogram FER", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Confusion matrix saved to: {save_path}")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """Plot One-vs-Rest ROC curves for each emotion class."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls_name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest) — Spectrogram FER", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"ROC curves saved to: {save_path}")


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    """t-SNE visualization of learned feature space."""
    log.info("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = labels == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   label=cls_name, color=color, alpha=0.7, s=30, edgecolors="none")

    ax.set_title("t-SNE of CNN Feature Embeddings — Spectrogram FER", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.2)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"t-SNE plot saved to: {save_path}")


# ─── Main Evaluation ──────────────────────────────────────────────────────────
def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded model: {model_name} (Val Acc: {checkpoint['val_acc']:.2f}%)")

    # Load validation data
    val_dataset = SpectrogramDataset(VAL_SPEC_DIR, transform=get_val_transform())
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Predictions
    y_true, y_pred, y_proba = get_predictions(model, val_loader, device)

    # ── Classification Report ──────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(report)

    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved to: {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, y_proba, REPORTS_DIR / "roc_curves.png")

    # t-SNE (only if model supports embedding extraction)
    embeddings, emb_labels = extract_embeddings(model, val_loader, device)
    if len(embeddings) > 0:
        plot_tsne(embeddings, emb_labels, REPORTS_DIR / "tsne_embeddings.png")

    log.info("Evaluation complete. Figures saved to reports/figures/")


if __name__ == "__main__":
    evaluate()
