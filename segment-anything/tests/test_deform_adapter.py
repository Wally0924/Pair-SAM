import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.deform_adapter import DeformAdapter


def _adapter(dim=32, h=4):
    a = DeformAdapter(vit_dim=dim, l2_channels=8, l3_channels=16, n_heads=4)
    feats = {'l2': torch.randn(1, 8, h * 2, h * 2),
             'l3': torch.randn(1, 16, h, h),
             'mask': torch.rand(1, 1, h * 2, h * 2)}
    a.set_features(feats, h, h)
    return a, h, dim


def test_block_indices():
    a = DeformAdapter(vit_dim=32, l2_channels=8, l3_channels=16, n_heads=4)
    assert a.INJECT_BLOCKS == [0, 8, 16, 24]
    assert a.EXTRACT_BLOCKS == [7, 15, 23]
    assert len(a.injectors) == 4 and len(a.extractors) == 3


def test_hooks_passthrough_when_no_features():
    """未呼叫 set_features（_c is None）時，hook 必須原樣放行、不崩潰。
    對應 forward 走 image_embedding 預算路徑、或 adapter 停用時。"""
    a = DeformAdapter(vit_dim=32, l2_channels=8, l3_channels=16, n_heads=4)
    x = torch.randn(1, 4, 4, 32)
    pre = a._make_inject_pre_hook(0)(None, (x,))
    assert pre is None or torch.equal(pre[0], x)   # 不修改輸入
    post = a._make_extract_post_hook(0)(None, None, x)
    assert torch.equal(post, x)


def test_inject_pre_hook_shape_preserved():
    a, h, dim = _adapter()
    hook = a._make_inject_pre_hook(0)
    x = torch.randn(1, h, h, dim)                      # SAM block I/O: (B,H,W,C)
    out = hook(None, (x,))
    assert isinstance(out, tuple) and out[0].shape == (1, h, h, dim)


def test_extract_post_hook_updates_c_returns_output_unchanged():
    a, h, dim = _adapter()
    c_before = a._c.clone()
    x = torch.randn(1, h, h, dim)
    hook = a._make_extract_post_hook(0)
    out = hook(None, None, x)
    assert torch.equal(out, x), "post-hook 必須回傳原 output 不變"
    assert not torch.equal(a._c, c_before), "c 必須被 extractor 更新"


def test_full_four_stage_sequence_runs():
    a, h, dim = _adapter()
    for s, blk in enumerate([0, 8, 16, 24]):
        x = torch.randn(1, h, h, dim)
        a._make_inject_pre_hook(s)(None, (x,))
        if s < 3:
            a._make_extract_post_hook(s)(None, None, torch.randn(1, h, h, dim))
    assert a._last_gate_val > 0.0
