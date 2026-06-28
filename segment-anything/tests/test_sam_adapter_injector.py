# segment-anything/tests/test_sam_adapter_injector.py
"""
測試 SameImageAdapterInjector（實驗 B：SAM-Adapter 同影像基線注入器）。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_sam_adapter_injector.py -v
"""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.modeling.sam_adapter_injector import SameImageAdapterInjector
from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_shape_and_residual():
    inj = SameImageAdapterInjector(vit_dim=1280, bottleneck=64)
    x = torch.randn(2, 8, 8, 1280)
    out = inj._inject_at_stage(x, 0)
    assert out.shape == x.shape


def test_no_reference_needed():
    """set_features 接收後忽略；注入必須在無任何參考下運作。"""
    inj = SameImageAdapterInjector(vit_dim=1280, bottleneck=64)
    inj.set_features({})            # 無 'l2'/'l3'/'mask' 鍵
    x = torch.randn(1, 4, 4, 1280)
    out = inj._inject_at_stage(x, 0)
    assert out.shape == x.shape
    assert inj.use_reference is False


def test_gate_init_is_005():
    inj = SameImageAdapterInjector(vit_dim=1280, bottleneck=64)
    g = F.softplus(inj.gates[0]).item()
    assert abs(g - 0.05) < 5e-3, f"expected ≈0.05, got {g:.4f}"


def test_param_count_ge_reference():
    """公平性：基線可訓練參數須 ≥ 參考注入器（避免低分歸咎於參數不足）。"""
    ref = MultiScaleCrossAttnInjector(vit_dim=1280, l2_channels=256,
                                      l3_channels=512, d_attn=256,
                                      pool_size=32, num_heads=4)
    base = SameImageAdapterInjector(vit_dim=1280, bottleneck=1700)
    assert _trainable(base) >= _trainable(ref), \
        f"baseline {_trainable(base)} < reference {_trainable(ref)}"


def test_delta_depends_on_input():
    """delta 由 ViT token 自身驅動（同影像 adapter）；梯度應回傳到輸入。"""
    inj = SameImageAdapterInjector(vit_dim=1280, bottleneck=64)
    torch.nn.init.normal_(inj.adapters[0][-1].weight, std=0.05)  # 讓 up-proj 非零
    x = torch.randn(1, 4, 4, 1280, requires_grad=True)
    out = inj._inject_at_stage(x, 0)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_hook_interface_parity():
    """必須具備與參考注入器一致的 hook 介面與屬性，供 WeatherSAM 多型呼叫。"""
    inj = SameImageAdapterInjector()
    assert inj.INJECT_BLOCKS == [7, 15, 23, 31]
    assert hasattr(inj, 'gates') and len(inj.gates) == 4
    assert hasattr(inj, '_make_pre_hook') and hasattr(inj, '_make_hook')
    assert hasattr(inj, 'set_features')
    for attr in ('_last_inject_cos_sim', '_last_gate_val',
                 '_last_delta_norm_ratio', '_last_kv_keep_ratio'):
        assert hasattr(inj, attr)
