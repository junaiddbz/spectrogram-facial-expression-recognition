"""
train.py
────────
Training pipeline for the Spectrogram FER model.

Features:
  - Two-stage training: frozen backbone → full fine-tune (for transfer learning)
  - CosineAnnealingLR learning rate scheduler
  - Mixed-precision training (torch.cuda.amp)
  - Early stopping based on validation accuracy
  - TensorBoard logging
  - Best model checkpoint saving

Usage:
    python src/train.py
    python src/train.py --model resnet18 --epochs 50 --batch_size 32
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_INTERVAL,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODEL_NAME,
    MODELS_DIR,
    NUM_EPOCHS,
    ONNX_MODEL_PATH,
    RANDOM_SEED,
    RUNS_DIR,
    WEIGHT_DECAY,
)
from dataset import get_dataloaders
from model import ModelType, build_model, export_to_onnx

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── One Epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """
    Run a single train or validation epoch.

    Returns:
        (avg_loss, accuracy_pct)
    """
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="Train" if is_train else "Val ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ─── Training Loop ────────────────────────────────────────────────────────────
def train(
    model_name: ModelType = MODEL_NAME,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_name = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))

    scaler = GradScaler(enabled=device.type == "cuda")
    best_val_acc = 0.0
    patience_counter = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: Freeze backbone, train head only (only for transfer learning models)
    # ══════════════════════════════════════════════════════════════════════════
    STAGE1_EPOCHS = 5
    if hasattr(model, "freeze_backbone") and model_name != "custom_cnn":
        log.info(f"─── Stage 1: Training head only for {STAGE1_EPOCHS} epochs ───")
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * 5,  # Higher LR for head-only training
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

        for epoch in range(1, STAGE1_EPOCHS + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
            val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      scaler, device, is_train=False)
            scheduler.step()

            log.info(f"[S1 Ep {epoch:02d}/{STAGE1_EPOCHS}] "
                     f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || "
                     f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        model.unfreeze_all()

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Full fine-tuning / training
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"─── Stage 2: Full training for up to {epochs} epochs ───")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      scaler, device, is_train=False)
        scheduler.step()

        # TensorBoard logging
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        log.info(f"[Ep {epoch:03d}/{epochs}] "
                 f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% || "
                 f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # ── Best model checkpoint ──────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_name":   model_name,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_acc":      val_acc,
                "val_loss":     val_loss,
            }, BEST_MODEL_PATH)
            log.info(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # ── Periodic checkpoint ───────────────────────────────────────────
        if epoch % CHECKPOINT_INTERVAL == 0:
            ckpt_path = MODELS_DIR / f"checkpoint_ep{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)

        # ── Early stopping ────────────────────────────────────────────────
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            log.info(f"Early stopping triggered after {epoch} epochs. "
                     f"Best Val Acc: {best_val_acc:.2f}%")
            break

    writer.close()
    log.info(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    log.info(f"Best model saved at: {BEST_MODEL_PATH}")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    log.info("Exporting best model to ONNX...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    export_to_onnx(model, str(ONNX_MODEL_PATH), device)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Spectrogram FER model.")
    parser.add_argument("--model",      type=str, default=MODEL_NAME,  help="Model architecture name")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS,   help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
