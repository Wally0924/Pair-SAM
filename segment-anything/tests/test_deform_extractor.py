import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import (
    Extractor, ReferencePriorModule, deform_inputs)


def _setup(dim=32, h=4):
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, l4_channels=16, dim=dim)
    c, _ = rpm({'l2': torch.randn(1, 8, h * 2, h * 2),
                'l3': torch.randn(1, 16, h, h),
                'l4': torch.randn(1, 16, h // 2, h // 2)})
    _, ext_in = deform_inputs(h, h, torch.device('cpu'))
    scale_hw = [(h * 2, h * 2), (h, h), (h // 2, h // 2)]
    x = torch.randn(1, h * h, dim)
    return Extractor(dim=dim, n_heads=4, deform_ratio=0.5), c, x, ext_in, scale_hw


def test_extract_updates_c_shape_preserved():
    ext, c, x, ext_in, scale_hw = _setup()
    c2 = ext(c, x, ext_in, scale_hw)
    assert c2.shape == c.shape and torch.isfinite(c2).all()


def test_vit_feat_grad_flows_to_vit():
    """端到端設計：K/V=ViT（不 detach）→ ∂sum(c')/∂x 應非 None 且非全 0。"""
    ext, c, _, ext_in, scale_hw = _setup()
    x = torch.randn(1, 16, 32, requires_grad=True)
    ext(c, x, ext_in, scale_hw).sum().backward()
    assert x.grad is not None and x.grad.abs().max().item() > 0.0


def test_c_receives_gradient():
    ext, c, x, ext_in, scale_hw = _setup()
    c = c.clone().requires_grad_(True)
    c.retain_grad()
    ext(c, x, ext_in, scale_hw).sum().backward()
    assert c.grad is not None and c.grad.abs().max().item() > 0.0
