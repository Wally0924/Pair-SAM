import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import ReferencePriorModule


def _feats(B=1, H16=4):
    H8 = H16 * 2
    return {'l2': torch.randn(B, 8, H8, H8),
            'l3': torch.randn(B, 16, H16, H16),
            'mask': torch.rand(B, 1, H8, H8)}


def test_c_shape_three_scales_concat():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    c, conf = rpm(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2           # 1/8 + 1/16 + 1/32
    assert c.shape == (1, L, 32)
    assert conf.shape == (1, L, 1)


def test_level_embed_makes_scales_distinguishable():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    assert rpm.level_embed.shape == (3, 32)


def test_use_reference_false_zeros_c_keeps_shape():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32, use_reference=False)
    c, conf = rpm(_feats(H16=4))
    L = 8 * 8 + 4 * 4 + 2 * 2
    assert c.shape == (1, L, 32)
    assert c.abs().max().item() == 0.0


def test_conf_all_ones_without_mask():
    rpm = ReferencePriorModule(l2_channels=8, l3_channels=16, dim=32)
    feats = _feats(H16=4); del feats['mask']
    c, conf = rpm(feats)
    assert torch.allclose(conf, torch.ones_like(conf))
