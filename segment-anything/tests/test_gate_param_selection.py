"""Tests for the _is_gate_param helper in weather_trainer.

Gate warmup 只涵蓋 legacy adapter 的 softplus gates。DeformAdapter (A3) 的
per-channel gamma 為零初始化（ViT-Adapter 原設定），零初始化本身即內建 warmup，
凍結在 0 反而使 attention 前幾個 epoch 收不到梯度，因此必須被排除。
"""
import os
import sys

# Add segment-anything root to path so we can import weather_trainer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import _is_gate_param from the real trainer module.
# weather_trainer.py has heavy module-level imports (WeatherSAM, loss utils).
# If those blow up (no GPU / missing deps), we want an explicit ImportError
# rather than silently duplicating the predicate.
# ---------------------------------------------------------------------------
from weather_trainer import _is_gate_param


# ---------------------------------------------------------------------------
# Predicate unit tests
# ---------------------------------------------------------------------------

class TestIsGateParamPredicate:

    # --- should return True (legacy softplus gates need warmup freeze) ---

    def test_legacy_gates_0(self):
        """Legacy MultiScaleCrossAttnInjector / SameImageAdapterInjector ParameterList."""
        assert _is_gate_param('vgg_injector.gates.0') is True

    def test_legacy_gates_3(self):
        assert _is_gate_param('vgg_injector.gates.3') is True

    # --- should return False ---

    def test_deform_adapter_gamma_excluded(self):
        """DeformAdapter gamma 零初始化，凍結它會讓 attention 收不到梯度 → 排除。"""
        assert _is_gate_param('vgg_injector.injectors.0.gamma') is False

    def test_deform_adapter_gamma_3_excluded(self):
        assert _is_gate_param('vgg_injector.injectors.3.gamma') is False

    def test_k_projs_weight(self):
        assert _is_gate_param('vgg_injector.k_projs.0.weight') is False

    def test_image_encoder_block(self):
        assert _is_gate_param('image_encoder.blocks.0.norm1.weight') is False

    def test_injectors_non_gate_weight(self):
        assert _is_gate_param('vgg_injector.injectors.0.attn.value_proj.weight') is False

    def test_injectors_attn_in_proj(self):
        assert _is_gate_param('vgg_injector.injectors.2.attn.in_proj.weight') is False


# ---------------------------------------------------------------------------
# Integration test: build a real DeformAdapter, attach it as `vgg_injector`
# on a throwaway Module, and confirm NO params are captured by the warmup.
# ---------------------------------------------------------------------------

class TestDeformAdapterGateCapture:

    def test_no_deform_adapter_params_captured(self):
        from segment_anything.modeling.deform_adapter import DeformAdapter

        # Small dims to keep the test lightweight (no GPU needed for param scan)
        adapter = DeformAdapter(
            vit_dim=64,       # small embedding dim
            l2_channels=32,
            l3_channels=64,
            l4_channels=64,
            n_heads=2,
            deform_ratio=0.5,
        )

        # Wrap in a parent module the same way WeatherSAM does
        parent = nn.Module()
        parent.vgg_injector = adapter

        captured = [n for n, _ in parent.named_parameters() if _is_gate_param(n)]
        assert captured == [], (
            f"DeformAdapter params must be excluded from gate warmup, got: {captured}"
        )

    def test_gamma_params_exist_and_zero_init(self):
        """gamma 仍存在（4 個 injector 各一個 per-channel 向量）且零初始化。"""
        from segment_anything.modeling.deform_adapter import DeformAdapter

        adapter = DeformAdapter(vit_dim=64, l2_channels=32, l3_channels=64,
                                l4_channels=64, n_heads=2)
        gammas = [(n, p) for n, p in adapter.named_parameters()
                  if n.endswith('.gamma')]
        assert len(gammas) == 4
        for n, p in gammas:
            assert p.shape == (64,), f"{n} 應為 per-channel 向量"
            assert p.abs().max().item() == 0.0, f"{n} 應零初始化"
