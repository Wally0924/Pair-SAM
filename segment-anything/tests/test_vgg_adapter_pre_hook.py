# segment-anything/tests/test_vgg_adapter_pre_hook.py
"""
測試 MultiScaleCrossAttnInjector v5（Cross-Attention，Q=ViT.detach()，全維，Xavier init）
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _small():
    return MultiScaleCrossAttnInjector(
        vit_dim=64, l2_channels=16, l3_channels=32,
        d_attn=32, pool_size=4, num_heads=4,
    )


def test_no_mlp_modules():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'vgg_mlp_downs'), "v4 MLP module must not exist in v5"
    assert not hasattr(inj, 'vgg_mlp_ups'),   "v4 MLP module must not exist in v5"


def test_has_cross_attn_modules():
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, 'k_projs')     and len(inj.k_projs)     == 4
    assert hasattr(inj, 'v_projs')     and len(inj.v_projs)     == 4
    assert hasattr(inj, 'cross_attns') and len(inj.cross_attns) == 4


def test_no_q_bottleneck():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'q_down_projs'), "Q must not have bottleneck projection in v5"


def test_gate_initial_value_approx_0_05():
    inj = MultiScaleCrossAttnInjector()
    gate = F.softplus(inj.gates[0])
    assert abs(gate.item() - 0.05) < 0.005, f"expected ≈0.05, got {gate.item():.4f}"


def test_xavier_init_not_zero():
    inj = MultiScaleCrossAttnInjector()
    for i in range(4):
        assert inj.k_projs[i].weight.abs().max().item() > 0.0, \
            f"k_projs[{i}] must be Xavier-init, not zero"
        assert inj.v_projs[i].weight.abs().max().item() > 0.0, \
            f"v_projs[{i}] must be Xavier-init, not zero"


def test_inject_shape_preserved():
    inj = _small()
    B, H, W, C = 2, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)


def test_delta_driven_by_vgg_not_vit():
    """固定 Q（ViT token），改變 VGG K/V → output 應改變。"""
    inj = _small()
    with torch.no_grad():
        for g in inj.gates:
            g.fill_(5.0)
    B, H, W, C = 1, 8, 8, 64
    vit = torch.randn(B, H, W, C)

    inj.set_features({'l2': torch.ones(B, 16, H, W), 'l3': torch.ones(B, 32, H, W)})
    out_a = inj._inject_at_stage(vit.clone(), 0)

    inj.set_features({'l2': -torch.ones(B, 16, H, W), 'l3': -torch.ones(B, 32, H, W)})
    out_b = inj._inject_at_stage(vit.clone(), 0)

    assert (out_a - out_b).abs().max().item() > 0.01, \
        "Different VGG feats must produce different output"


def test_vit_q_detached_no_grad():
    """Q 必須 detach：梯度只走殘差路徑，vit_input.grad 應為全 1。
    原理：out = q + gate*delta；若 Q 正確 detach，delta 對 vit_input 無梯度，
    ∂sum(out)/∂vit_input = 1（all-ones）。若未 detach 則 grad ≠ ones。
    """
    inj = _small()
    B, H, W, C = 1, 4, 4, 64
    vit_input = torch.randn(B, H, W, C, requires_grad=True)
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(vit_input, 0)
    out.sum().backward()
    assert vit_input.grad is not None
    assert torch.allclose(vit_input.grad, torch.ones_like(vit_input)), \
        "grad must be all-ones (residual only); non-ones means Q is NOT detached"


def test_diagnostics_updated_after_all_stages():
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    for i in range(4):
        inj._inject_at_stage(torch.randn(B, H, W, C), i)
    assert not math.isnan(inj._last_inject_cos_sim)
    assert inj._last_gate_val > 0.0
    assert inj._last_delta_norm_ratio >= 0.0


def test_pre_hook_returns_tuple_of_correct_shape():
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    hook = inj._make_pre_hook(0)
    result = hook(nn.Linear(1, 1), (torch.randn(B, H, W, C),))
    assert isinstance(result, tuple) and len(result) == 1
    assert result[0].shape == (B, H, W, C)


def test_make_hook_post_still_exists():
    """_make_hook (post-hook) 必須保留供 ablation 使用。"""
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, '_make_hook')


def test_mask_aware_path_activated_when_mask_provided():
    """提供 mask 時應啟用 confidence-aware path，更新 _last_kv_keep_ratio。"""
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, :W // 2] = 1.0
    feats = {
        'l2':   torch.randn(B, 16, H, W) * mask,
        'l3':   torch.randn(B, 32, H, W) * mask,
        'mask': mask,
    }
    inj.set_features(feats)
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)
    assert 0.0 < inj._last_kv_keep_ratio < 1.0, \
        f"expected partial keep_ratio, got {inj._last_kv_keep_ratio:.4f}"


def test_backward_compat_no_mask_key():
    """未提供 mask 時應退回 v5 原始邏輯，_last_kv_keep_ratio 維持 1.0。"""
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)
    assert inj._last_kv_keep_ratio == 1.0


def test_all_masked_fallback_no_nan():
    """全 mask=0 的極端情況：fallback 保留最高 valid_ratio 的 cell，不可 NaN。"""
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    feats = {
        'l2':   torch.zeros(B, 16, H, W),
        'l3':   torch.zeros(B, 32, H, W),
        'mask': torch.zeros(B, 1,  H, W),
    }
    inj.set_features(feats)
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert torch.isfinite(out).all(), "output must be finite even when mask is all zero"
