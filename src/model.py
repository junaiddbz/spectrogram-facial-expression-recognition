"""
model.py
────────
CNN Model Definitions for Spectrogram-based Facial Expression Recognition.

Three architectures are implemented for comparison:
    1. CustomCNN        — Lightweight 4-block CNN, built from scratch
    2. ResNet18FER      — ResNet-18 with custom classification head (recommended)
    3. MobileNetV3FER   — MobileNetV3-Small for lightweight inference
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights

from config import (
    DROPOUT_RATE,
    FC_HIDDEN_DIM,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_CLASSES,
    PRETRAINED,
)

log = logging.getLogger(__name__)

ModelType = Literal["resnet18", "mobilenet_v3_small", "custom_cnn"]


# ─── 1. Custom CNN ─────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Lightweight 4-block CNN trained from scratch.
    Input:  (B, 3, 224, 224)
    Output: (B, NUM_CLASSES)
    """

    def __init__(
        self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(3, 32),  # → (B, 32, 112, 112)
            _ConvBlock(32, 64),  # → (B, 64,  56,  56)
            _ConvBlock(64, 128),  # → (B, 128, 28,  28)
            _ConvBlock(128, 256),  # → (B, 256, 14,  14)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → (B, 256, 4, 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─── 2. ResNet-18 (Transfer Learning) ─────────────────────────────────────────
class ResNet18FER(nn.Module):
    """
    ResNet-18 with custom classification head, fine-tuned for FER.
    Uses ImageNet pretrained weights by default.

    Input:  (B, 3, 224, 224)
    Output: (B, NUM_CLASSES)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_DIM, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the custom head (for feature extraction phase)."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        log.info("Backbone frozen. Only classification head will be trained.")

    def unfreeze_all(self) -> None:
        """Unfreeze all layers (for fine-tuning phase)."""
        for param in self.model.parameters():
            param.requires_grad = True
        log.info("All layers unfrozen for full fine-tuning.")


# ─── 3. MobileNetV3-Small (Lightweight) ───────────────────────────────────────
class MobileNetV3FER(nn.Module):
    """
    MobileNetV3-Small with custom head for fast CPU inference on Hugging Face Spaces.

    Input:  (B, 3, 224, 224)
    Output: (B, NUM_CLASSES)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        in_features = backbone.classifier[-1].in_features  # 1024
        backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_model(model_name: ModelType = "resnet18") -> nn.Module:
    """
    Instantiate and return a model by name.

    Args:
        model_name: One of "resnet18", "mobilenet_v3_small", "custom_cnn"

    Returns:
        Initialized nn.Module
    """
    model_map = {
        "resnet18": ResNet18FER,
        "mobilenet_v3_small": MobileNetV3FER,
        "custom_cnn": CustomCNN,
    }
    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: '{model_name}'. Choose from {list(model_map.keys())}"
        )

    model = model_map[model_name]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Built model: {model_name} | Trainable params: {n_params:,}")
    return model


def export_to_onnx(model: nn.Module, output_path: str, device: torch.device) -> None:
    """
    Export a trained PyTorch model to ONNX format for fast CPU inference.

    Args:
        model:       Trained nn.Module in eval mode
        output_path: Path to save the .onnx file
        device:      torch.device to run the dummy forward pass on
    """
    model.eval()
    dummy_input = torch.randn(1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Opset 17 supported by onnxruntime 1.17.0
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    log.info(f"ONNX model exported to: {output_path}")
