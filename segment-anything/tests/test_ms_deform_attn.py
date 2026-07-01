"""pure-PyTorch MSDeformAttn 去風險測試（無 CUDA 編譯）。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_ms_deform_attn.py -v
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.ops.ms_deform_attn import MSDeformAttn


def _ref_points(spatial_shapes, device):
    refs = []
    for (H, W) in spatial_shapes:
        ry, rx = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, device=device) / W,
            indexing='ij')
        refs.append(torch.stack((rx.reshape(-1), ry.reshape(-1)), -1))
    return torch.cat(refs, 0)[None, :, None, :]  # (1, Lq, 1, 2)


def test_forward_shape_preserved():
    d, N = 32, 2
    shapes = [(8, 8), (4, 4)]
    lens = [h * w for h, w in shapes]
    lsi = torch.tensor([0, lens[0]])
    attn = MSDeformAttn(d_model=d, n_levels=len(shapes), n_heads=4, n_points=4, ratio=0.5)
    value = torch.randn(N, sum(lens), d)
    query = torch.randn(N, lens[0], d)  # query on level-0 grid
    ref = _ref_points([shapes[0]], value.device).expand(N, -1, -1, -1)
    out = attn(query, ref, value, torch.tensor(shapes), lsi)
    assert out.shape == (N, lens[0], d)
    assert torch.isfinite(out).all()


def test_gradient_flows_to_value_and_params():
    d, N = 32, 1
    shapes = [(8, 8)]
    lens = [h * w for h, w in shapes]
    attn = MSDeformAttn(d_model=d, n_levels=1, n_heads=4, n_points=4, ratio=0.5)
    value = torch.randn(N, sum(lens), d, requires_grad=True)
    query = torch.randn(N, lens[0], d)
    ref = _ref_points(shapes, value.device).expand(N, -1, -1, -1)
    attn(query, ref, value, torch.tensor(shapes), torch.tensor([0])).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert attn.sampling_offsets.weight.grad is not None
