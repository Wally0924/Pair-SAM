import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import ReferencePriorModule


def _feats(B=1, H16=4):
    H8, H32 = H16 * 2, H16 // 2
    return {'l2': torch.randn(B, 8, H8, H8),
            'l3': torch.randn(B, 16, H16, H16),
            'l4': torch.randn(B, 16, H32, H32),
            'mask': torch.rand(B, 1, H8, H8)}


def _rpm(**kw):
    return ReferencePriorModule(l2_channels=8, l3_channels=16, l4_channels=16,
                                dim=32, **kw)


def test_c_shape_three_scales_concat():
    c, conf = _rpm()(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2           # 1/8 + 1/16 + 1/32
    assert c.shape == (1, L, 32)
    assert conf.shape == (1, L, 1)


def test_level_embed_makes_scales_distinguishable():
    assert _rpm().level_embed.shape == (3, 32)


def test_use_reference_false_zeros_c_keeps_shape():
    c, conf = _rpm(use_reference=False)(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2
    assert c.shape == (1, L, 32)
    assert c.abs().max().item() == 0.0


def test_conf_all_ones_without_mask():
    feats = _feats(H16=4); del feats['mask']
    c, conf = _rpm()(feats)
    assert torch.allclose(conf, torch.ones_like(conf))


def test_scale_1_32_comes_from_l4_not_l3():
    """真多尺度驗證：c 的 1/32 段只由 l4 決定；改 l3 不得影響該段。"""
    rpm = _rpm()
    feats = _feats(H16=4)
    c0, _ = rpm(feats)
    seg = 8 * 8 + 4 * 4                  # 1/32 段起點
    c1, _ = rpm({**feats, 'l3': torch.randn_like(feats['l3'])})
    assert torch.allclose(c0[:, seg:], c1[:, seg:]), "l3 改變不應影響 1/32 段"
    c2, _ = rpm({**feats, 'l4': torch.randn_like(feats['l4'])})
    assert not torch.allclose(c0[:, seg:], c2[:, seg:]), "l4 改變必須反映在 1/32 段"
