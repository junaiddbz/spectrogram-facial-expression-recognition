"""
dataset.py
──────────
PyTorch Dataset class for loading spectrogram images from disk.
Applies train-time augmentations and normalization for pretrained CNNs.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    AUG_COLOR_JITTER,
    AUG_HORIZONTAL_FLIP_P,
    AUG_ROTATION_DEGREES,
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_SPEC_DIR,
    VAL_SPEC_DIR,
)

log = logging.getLogger(__name__)


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading spectrogram PNG images.

    Directory structure expected:
        root_dir/
          {emotion_class}/
            *.png

    Args:
        root_dir:    Path to split directory (e.g., .../spectrograms/train)
        transform:   torchvision transform pipeline
        class_names: Ordered list of emotion class names (must match directory names)
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        class_names: list = CLASS_NAMES,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(class_names)}

        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Walk root_dir and collect (image_path, label) tuples."""
        found_classes = set()
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            if class_name not in self.class_to_idx:
                log.warning(f"Unrecognized class directory: {class_dir.name}. Skipping.")
                continue
            label = self.class_to_idx[class_name]
            found_classes.add(class_name)
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.root_dir}. " "Run make_spectrograms.py first."
            )

        log.info(
            f"Loaded {len(self.samples)} samples from {self.root_dir} "
            f"({len(found_classes)} classes)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self) -> dict:
        """Return count of samples per class (for imbalance analysis)."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


# ─── Transform Pipelines ──────────────────────────────────────────────────────
def get_train_transform() -> transforms.Compose:
    """Augmented transform for training split."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=AUG_HORIZONTAL_FLIP_P),
            transforms.ColorJitter(**AUG_COLOR_JITTER),
            transforms.RandomRotation(degrees=AUG_ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation/inference."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


# ─── DataLoader Factory ───────────────────────────────────────────────────────
def get_dataloaders(
    train_dir: Path = TRAIN_SPEC_DIR,
    val_dir: Path = VAL_SPEC_DIR,
    batch_size: int = BATCH_SIZE,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        train_dir:            Path to training spectrogram directory.
        val_dir:              Path to validation spectrogram directory.
        batch_size:           Batch size for both loaders.
        use_weighted_sampler: If True, uses WeightedRandomSampler to handle
                              class imbalance (recommended for CK+).

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SpectrogramDataset(train_dir, transform=get_train_transform())
    val_dataset = SpectrogramDataset(val_dir, transform=get_val_transform())

    train_sampler = None
    if use_weighted_sampler:
        counts = train_dataset.class_counts
        total = len(train_dataset)
        weights_per_class = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
        sample_weights = [weights_per_class[label] for _, label in train_dataset.samples]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )
        log.info("Using WeightedRandomSampler to handle class imbalance.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    log.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    return train_loader, val_loader
