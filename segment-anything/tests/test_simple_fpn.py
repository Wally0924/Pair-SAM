import torch
import pytest

from segment_anything.modeling.simple_fpn import SimpleFPN


def test_output_shapes():
    fpn = SimpleFPN(dim=256)
    x = torch.randn(1, 256, 64, 64)
    feats, mask_features = fpn(x)
    assert [tuple(f.shape) for f in feats] == [
        (1, 256, 32, 32),   # 1/32
        (1, 256, 64, 64),   # 1/16
        (1, 256, 128, 128), # 1/8
    ]
    assert tuple(mask_features.shape) == (1, 256, 256, 256)  # 1/4


def test_gradient_flows_to_input():
    fpn = SimpleFPN(dim=256)
    x = torch.randn(1, 256, 64, 64, requires_grad=True)
    feats, mask_features = fpn(x)
    (sum(f.sum() for f in feats) + mask_features.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fp16_autocast_forward():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for autocast test")
    fpn = SimpleFPN(dim=256).cuda()
    x = torch.randn(1, 256, 64, 64, device="cuda")
    with torch.amp.autocast("cuda"):
        feats, mask_features = fpn(x)
    assert torch.isfinite(mask_features).all()
