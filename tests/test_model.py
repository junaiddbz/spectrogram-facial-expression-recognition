"""
test_model.py
─────────────
Unit tests for model architectures and export.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES  # noqa: E402
from model import CustomCNN, MobileNetV3FER, ResNet18FER, build_model  # noqa: E402

DUMMY_INPUT = torch.randn(2, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)


class TestCustomCNN:
    def test_output_shape(self):
        model = CustomCNN(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (
            2,
            NUM_CLASSES,
        ), f"Expected (2, {NUM_CLASSES}), got {out.shape}"

    def test_no_nan_in_output(self):
        model = CustomCNN()
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert not torch.isnan(out).any(), "Output contains NaN"


class TestResNet18FER:
    def test_output_shape(self):
        model = ResNet18FER(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (2, NUM_CLASSES)

    def test_freeze_unfreeze(self):
        model = ResNet18FER(pretrained=False)
        model.freeze_backbone()
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen > 0, "No parameters were frozen"

        model.unfreeze_all()
        unfrozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert unfrozen == 0, "Parameters still frozen after unfreeze_all()"


class TestMobileNetV3FER:
    def test_output_shape(self):
        model = MobileNetV3FER(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert out.shape == (2, NUM_CLASSES)


class TestBuildModel:
    def test_build_resnet18(self):
        model = build_model("resnet18")
        assert isinstance(model, ResNet18FER)

    def test_build_custom_cnn(self):
        model = build_model("custom_cnn")
        assert isinstance(model, CustomCNN)

    def test_build_mobilenet(self):
        model = build_model("mobilenet_v3_small")
        assert isinstance(model, MobileNetV3FER)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("unknown_architecture")
