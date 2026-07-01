"""TDD test for _is_gate_param helper in weather_trainer.

RED phase: tests are written before the fix exists.
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

    # --- should return True ---

    def test_deform_adapter_injector_0_gate(self):
        assert _is_gate_param('vgg_injector.injectors.0.gate') is True

    def test_deform_adapter_injector_3_gate(self):
        assert _is_gate_param('vgg_injector.injectors.3.gate') is True

    def test_legacy_gates_0(self):
        """Legacy MultiScaleCrossAttnInjector / SameImageAdapterInjector ParameterList."""
        assert _is_gate_param('vgg_injector.gates.0') is True

    def test_legacy_gates_3(self):
        assert _is_gate_param('vgg_injector.gates.3') is True

    # --- should return False ---

    def test_k_projs_weight(self):
        assert _is_gate_param('vgg_injector.k_projs.0.weight') is False

    def test_image_encoder_block(self):
        assert _is_gate_param('image_encoder.blocks.0.norm1.weight') is False

    def test_injectors_non_gate_weight(self):
        """A param whose name contains 'injectors' but is NOT the bare gate."""
        assert _is_gate_param('vgg_injector.injectors.0.attn.value_proj.weight') is False

    def test_injectors_attn_in_proj(self):
        assert _is_gate_param('vgg_injector.injectors.2.attn.in_proj.weight') is False


# ---------------------------------------------------------------------------
# Integration test: build a real DeformAdapter, attach it as `vgg_injector`
# on a throwaway Module, and confirm exactly 4 gate params are captured.
# ---------------------------------------------------------------------------

class TestDeformAdapterGateCapture:

    def test_exactly_four_gates_captured(self):
        from segment_anything.modeling.deform_adapter import DeformAdapter

        # Small dims to keep the test lightweight (no GPU needed for param scan)
        adapter = DeformAdapter(
            vit_dim=64,       # small embedding dim
            l2_channels=32,
            l3_channels=64,
            n_heads=2,
            deform_ratio=0.5,
        )

        # Wrap in a parent module the same way WeatherSAM does
        parent = nn.Module()
        parent.vgg_injector = adapter

        gate_params = [
            p for n, p in parent.named_parameters()
            if _is_gate_param(n)
        ]

        # DeformAdapter has 4 Injectors, each with one bare nn.Parameter gate
        assert len(gate_params) == 4, (
            f"Expected 4 gate params, got {len(gate_params)}. "
            f"Gate param names: {[n for n, p in parent.named_parameters() if _is_gate_param(n)]}"
        )

    def test_gate_param_names_are_correct(self):
        """Verify the actual names we capture look as expected."""
        from segment_anything.modeling.deform_adapter import DeformAdapter

        adapter = DeformAdapter(vit_dim=64, l2_channels=32, l3_channels=64, n_heads=2)
        parent = nn.Module()
        parent.vgg_injector = adapter

        captured_names = [n for n, _ in parent.named_parameters() if _is_gate_param(n)]
        expected_names = [
            'vgg_injector.injectors.0.gate',
            'vgg_injector.injectors.1.gate',
            'vgg_injector.injectors.2.gate',
            'vgg_injector.injectors.3.gate',
        ]
        assert captured_names == expected_names, (
            f"Captured: {captured_names}\nExpected: {expected_names}"
        )
