# segment-anything/tests/test_vgg_adapter_pre_hook.py
"""
測試 MultiScaleCrossAttnInjector v4（SAM-Adapter 風格 MLP + softplus gate）
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
        vit_dim=64, l2_channels=16, l3_channels=32, d_hidden=32, pool_size=4
    )


def test_no_cross_attention_modules():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'cross_attns')
    assert not hasattr(inj, 'q_down_projs')
    assert not hasattr(inj, 'q_up_projs')


def test_has_mlp_modules():
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, 'vgg_mlp_downs') and len(inj.vgg_mlp_downs) == 4
    assert hasattr(inj, 'vgg_mlp_ups')   and len(inj.vgg_mlp_ups)   == 4


def test_gate_initial_value_approx_0_05():
    inj = MultiScaleCrossAttnInjector()
    gate = F.softplus(inj.gates[0])
    assert abs(gate.item() - 0.05) < 0.005, f"expected ≈0.05, got {gate.item():.4f}"


def test_softplus_gradient_stronger_than_sigmoid():
    raw = torch.tensor(-2.9444, requires_grad=True)
    F.softplus(raw).backward()
    sp_grad = raw.grad.item()

    raw2 = torch.tensor(-5.0, requires_grad=True)
    torch.sigmoid(raw2).backward()
    sig_grad = raw2.grad.item()

    assert sp_grad > sig_grad * 5, f"softplus grad {sp_grad:.4f} should be >> sigmoid grad {sig_grad:.4f}"


def test_mlp_up_zero_initialized():
    inj = MultiScaleCrossAttnInjector()
    for i, proj in enumerate(inj.vgg_mlp_ups):
        assert proj.weight.abs().max().item() == 0.0, f"vgg_mlp_ups[{i}] not zero-init"


def test_inject_shape_preserved():
    inj = _small()
    B, H, W, C = 2, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)


def test_delta_driven_by_vgg_not_vit():
    """固定 ViT token，改變 VGG feats → output 應改變。"""
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

    assert (out_a - out_b).abs().max().item() > 0.01, "Different VGG feats must produce different output"


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
