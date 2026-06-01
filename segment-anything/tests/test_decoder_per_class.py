# segment-anything/tests/test_decoder_per_class.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_decoder_per_class.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from segment_anything.modeling.weather_mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer


def _make_decoder(num_classes=4):
    tf = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=512)
    dec = MaskDecoder(transformer_dim=256, transformer=tf, num_classes=num_classes)
    dec.eval()
    return dec


def _inputs(K=2):
    img = torch.randn(1, 256, 64, 64)
    pe = torch.randn(1, 256, 64, 64)
    sparse = torch.randn(K, 2, 256)      # K classes, N_tok=2
    dense = torch.randn(1, 256, 64, 64)
    class_ids = list(range(K))
    return img, pe, sparse, dense, class_ids


def test_default_mode_is_unified():
    dec = _make_decoder()
    assert dec.decoder_mode == 'unified'


def test_param_count_identical_across_modes():
    dec = _make_decoder()
    n = sum(p.numel() for p in dec.parameters())
    dec.decoder_mode = 'per_class'
    assert sum(p.numel() for p in dec.parameters()) == n


def test_per_class_isolates_classes_unified_does_not():
    dec = _make_decoder()
    img, pe, sparse, dense, class_ids = _inputs(K=2)
    sparse2 = sparse.clone()
    sparse2[1] += 5.0  # 只擾動 class 1 的 prompt

    with torch.no_grad():
        dec.decoder_mode = 'per_class'
        a = dec.forward_semantic(img, pe, sparse, dense, class_ids)
        b = dec.forward_semantic(img, pe, sparse2, dense, class_ids)
        assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)  # class 0 不受 class 1 影響

        dec.decoder_mode = 'unified'
        c = dec.forward_semantic(img, pe, sparse, dense, class_ids)
        d = dec.forward_semantic(img, pe, sparse2, dense, class_ids)
        assert not torch.allclose(c[:, 0], d[:, 0], atol=1e-5)  # unified：受影響
