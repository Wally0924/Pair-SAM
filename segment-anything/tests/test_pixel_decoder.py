import torch
import pytest

from segment_anything.modeling.msdeform_pixel_decoder import MSDeformAttnPixelDecoder


def _make_inputs(requires_grad=False):
    # SimpleFPN 輸出契約：feats=[f32,f16,f8]（coarse→fine），mask src=f4（1/4）
    f32 = torch.randn(1, 256, 32, 32, requires_grad=requires_grad)
    f16 = torch.randn(1, 256, 64, 64, requires_grad=requires_grad)
    f8 = torch.randn(1, 256, 128, 128, requires_grad=requires_grad)
    f4 = torch.randn(1, 256, 256, 256, requires_grad=requires_grad)
    return f32, f16, f8, f4


def test_output_shapes():
    dec = MSDeformAttnPixelDecoder()
    f32, f16, f8, f4 = _make_inputs()
    feats, mask_features = dec([f32, f16, f8], f4)
    # pixel decoder 只做跨尺度融合，對外形狀須與 SimpleFPN 三尺度逐一相同
    assert [tuple(f.shape) for f in feats] == [
        (1, 256, 32, 32),    # 1/32
        (1, 256, 64, 64),    # 1/16
        (1, 256, 128, 128),  # 1/8
    ]
    assert tuple(mask_features.shape) == (1, 256, 256, 256)  # 1/4


def test_gradient_flows_to_inputs():
    dec = MSDeformAttnPixelDecoder()
    f32, f16, f8, f4 = _make_inputs(requires_grad=True)
    feats, mask_features = dec([f32, f16, f8], f4)
    (sum(f.sum() for f in feats) + mask_features.sum()).backward()
    for t in (f32, f16, f8, f4):
        assert t.grad is not None and torch.isfinite(t.grad).all()
