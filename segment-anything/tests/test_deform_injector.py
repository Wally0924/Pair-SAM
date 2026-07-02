import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import Injector, ReferencePriorModule, deform_inputs


def _setup(dim=32, h=4):
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, l4_channels=16, dim=dim)
    feats = {'l2': torch.randn(1, 8, h * 2, h * 2),
             'l3': torch.randn(1, 16, h, h),
             'l4': torch.randn(1, 16, h // 2, h // 2),
             'mask': torch.rand(1, 1, h * 2, h * 2)}
    c, conf = rpm(feats)
    inj_in, _ = deform_inputs(h, h, torch.device('cpu'))
    x = torch.randn(1, h * h, dim)
    return Injector(dim=dim, n_heads=4, n_levels=3, deform_ratio=0.5), x, c, conf, inj_in


def test_inject_shape_preserved():
    inj, x, c, conf, inj_in = _setup()
    out = inj(x, c, conf, inj_in)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_gamma_zero_init_identity():
    """ViT-Adapter 原設定：per-channel gamma 零初始化 → 初始為恆等映射。"""
    inj, x, c, conf, inj_in = _setup()
    assert inj.gamma.shape == (32,)
    assert inj.gamma.abs().max().item() == 0.0
    out = inj(x, c, conf, inj_in)
    assert torch.equal(out, x), "gamma=0 時輸出必須嚴格等於輸入"


def test_query_detached_residual_only_grad():
    """Q detach → ∂sum(out)/∂x = 1（純殘差）。"""
    inj, _, c, conf, inj_in = _setup()
    x = torch.randn(1, 16, 32, requires_grad=True)
    inj(x, c, conf, inj_in).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), "Q 未 detach：grad 非全 1"


def test_low_confidence_weakens_injection():
    inj, x, c, conf, inj_in = _setup()
    with torch.no_grad():
        inj.gamma.fill_(5.0)
    out_hi = inj(x.clone(), c, torch.ones_like(conf), inj_in)
    out_lo = inj(x.clone(), c, torch.zeros_like(conf), inj_in)
    # conf=0 → value 全 0 → 注入量應明顯小於 conf=1
    assert (out_lo - x).abs().mean() < (out_hi - x).abs().mean()


def test_injection_uses_all_three_scales():
    """真多尺度驗證：單獨擾動任一尺度（含 1/32 的 l4 段）都必須改變注入輸出。"""
    inj, x, c, conf, inj_in = _setup(h=4)
    with torch.no_grad():
        inj.gamma.fill_(1.0)
    base = inj(x, c, conf, inj_in)
    n8, n16 = 8 * 8, 4 * 4
    segments = {'1/8': (0, n8), '1/16': (n8, n8 + n16), '1/32': (n8 + n16, c.shape[1])}
    for name, (s, e) in segments.items():
        c_mod = c.clone()
        c_mod[:, s:e] = torch.randn_like(c_mod[:, s:e]) * 10.0
        out = inj(x, c_mod, conf, inj_in)
        assert not torch.allclose(base, out), f"擾動 {name} 段未影響注入輸出"
