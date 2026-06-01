"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_ref_switch.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _make_injector():
    inj = MultiScaleCrossAttnInjector()
    inj.eval()
    return inj


def _feats(H=16, W=16):
    # l2: 256ch, l3: 512ch（符合 fusion.pre_align 輸出維度）；不放 'mask' key 走 no-mask 分支
    return {'l2': torch.randn(1, 256, H, W), 'l3': torch.randn(1, 512, H, W)}


def test_use_reference_default_true():
    assert _make_injector().use_reference is True


def test_ref_off_insensitive_to_reference_content():
    inj = _make_injector()
    inj.use_reference = False
    out = torch.randn(1, 16, 16, inj.vit_dim)
    with torch.no_grad():
        inj.set_features(_feats()); a = inj._inject_at_stage(out.clone(), 0)
        inj.set_features(_feats()); b = inj._inject_at_stage(out.clone(), 0)
    assert torch.allclose(a, b, atol=1e-5)   # ref off：與參考內容無關 → 兩次相同


def test_ref_on_sensitive_to_reference_content():
    inj = _make_injector()
    inj.use_reference = True
    out = torch.randn(1, 16, 16, inj.vit_dim)
    with torch.no_grad():
        inj.set_features(_feats()); a = inj._inject_at_stage(out.clone(), 0)
        inj.set_features(_feats()); b = inj._inject_at_stage(out.clone(), 0)
    assert not torch.allclose(a, b, atol=1e-5)  # ref on：不同參考 → 不同輸出
