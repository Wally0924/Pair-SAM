"""
Smoke test: verify enable_vgg_adapter(mode) wires hooks correctly without loading SAM weights.
Tests hook registration plumbing; functional correctness is covered in test_vgg_adapter_pre_hook.py.
"""
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


class FakeBlock(nn.Module):
    def forward(self, x):
        return x


class FakeEncoder(nn.Module):
    def __init__(self, n_blocks=32):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(n_blocks)])
        self.img_size = 1024

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FakePairSAM(nn.Module):
    """Minimal stand-in for PairSAM — only the adapter-related attributes."""
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeEncoder(n_blocks=32)
        self.vgg_injector = MultiScaleCrossAttnInjector(
            vit_dim=1280, l2_channels=256, l3_channels=512,
            d_attn=256, pool_size=32, num_heads=4,
        )
        self.use_vgg_adapter = False
        self._adapter_hook_handles = []

    # Copy the real enable/disable methods verbatim (no SAM-specific deps)
    def enable_vgg_adapter(self, mode: str = 'pre'):
        import warnings
        if mode not in ('pre', 'post'):
            raise ValueError(f"[PairSAM] mode must be 'pre' or 'post', got {mode!r}")
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []
        all_inject_blocks = self.vgg_injector.INJECT_BLOCKS
        n_blocks = len(self.image_encoder.blocks)
        inject_blocks = [b for b in all_inject_blocks if b < n_blocks]
        if len(inject_blocks) != len(all_inject_blocks):
            warnings.warn(f"[PairSAM] Some INJECT_BLOCKS out of range", stacklevel=2)
        for stage_idx, block_idx in enumerate(inject_blocks):
            target_block = self.image_encoder.blocks[block_idx]
            if mode == 'pre':
                handle = target_block.register_forward_pre_hook(
                    self.vgg_injector._make_pre_hook(stage_idx))
            else:
                handle = target_block.register_forward_hook(
                    self.vgg_injector._make_hook(stage_idx))
            self._adapter_hook_handles.append(handle)
        self.use_vgg_adapter = True

    def disable_vgg_adapter(self):
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []
        self.use_vgg_adapter = False


def test_enable_pre_mode_registers_four_hooks():
    model = FakePairSAM()
    model.enable_vgg_adapter(mode='pre')
    assert model.use_vgg_adapter is True
    assert len(model._adapter_hook_handles) == 4


def test_enable_post_mode_registers_four_hooks():
    model = FakePairSAM()
    model.enable_vgg_adapter(mode='post')
    assert model.use_vgg_adapter is True
    assert len(model._adapter_hook_handles) == 4


def test_disable_clears_handles():
    model = FakePairSAM()
    model.enable_vgg_adapter(mode='pre')
    model.disable_vgg_adapter()
    assert model.use_vgg_adapter is False
    assert len(model._adapter_hook_handles) == 0


def test_invalid_mode_raises_value_error():
    model = FakePairSAM()
    with pytest.raises(ValueError, match="mode must be 'pre' or 'post'"):
        model.enable_vgg_adapter(mode='invalid')


def test_re_enable_clears_old_hooks():
    model = FakePairSAM()
    model.enable_vgg_adapter(mode='pre')
    assert len(model._adapter_hook_handles) == 4
    model.enable_vgg_adapter(mode='post')  # re-enable with different mode
    assert len(model._adapter_hook_handles) == 4  # not 8


def test_pre_hook_fires_during_forward():
    """Verify pre-hook actually runs during forward pass (not just registered)."""
    model = FakePairSAM()
    model.vgg_injector.set_features({
        'l2': torch.zeros(1, 256, 64, 64),
        'l3': torch.zeros(1, 512, 64, 64),
    })
    model.enable_vgg_adapter(mode='pre')
    x = torch.zeros(1, 64, 64, 1280)
    with torch.no_grad():
        _ = model.image_encoder(x)
    assert model.vgg_injector._global_step > 0


def test_post_hook_fires_during_forward():
    """Verify post-hook actually runs during forward pass."""
    model = FakePairSAM()
    model.vgg_injector.set_features({
        'l2': torch.zeros(1, 256, 64, 64),
        'l3': torch.zeros(1, 512, 64, 64),
    })
    model.enable_vgg_adapter(mode='post')
    x = torch.zeros(1, 64, 64, 1280)
    with torch.no_grad():
        _ = model.image_encoder(x)
    assert model.vgg_injector._global_step > 0
