import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import Injector, ReferencePriorModule, deform_inputs


def _setup(dim=32, h=4):
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=dim)
    feats = {'l2': torch.randn(1, 8, h * 2, h * 2),
             'l3': torch.randn(1, 16, h, h),
             'mask': torch.rand(1, 1, h * 2, h * 2)}
    c, conf = rpm(feats)
    inj_in, _ = deform_inputs(h, h, torch.device('cpu'))
    x = torch.randn(1, h * h, dim)
    return Injector(dim=dim, n_heads=4, n_levels=3, deform_ratio=0.5), x, c, conf, inj_in


def test_inject_shape_preserved():
    inj, x, c, conf, inj_in = _setup()
    out = inj(x, c, conf, inj_in)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_gate_initial_value_approx_0_05():
    inj = Injector(dim=32, n_heads=4)
    assert abs(F.softplus(inj.gate).item() - 0.05) < 0.005


def test_query_detached_residual_only_grad():
    """Q detach → ∂sum(out)/∂x = 1（純殘差）。"""
    inj, _, c, conf, inj_in = _setup()
    x = torch.randn(1, 16, 32, requires_grad=True)
    inj(x, c, conf, inj_in).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), "Q 未 detach：grad 非全 1"


def test_low_confidence_weakens_injection():
    inj, x, c, conf, inj_in = _setup()
    with torch.no_grad():
        inj.gate.fill_(5.0)
    out_hi = inj(x.clone(), c, torch.ones_like(conf), inj_in)
    out_lo = inj(x.clone(), c, torch.zeros_like(conf), inj_in)
    # conf=0 → value 全 0 → 注入量應明顯小於 conf=1
    assert (out_lo - x).abs().mean() < (out_hi - x).abs().mean()
