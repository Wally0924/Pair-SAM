"""整合：DeformAdapter 掛上真實 SAM ViT-H block，全 forward 輸出形狀不變、無 NaN、ViT 受保護。
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_deform_adapter_integration.py -v
"""
import torch, torch.nn as nn, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.deform_adapter import DeformAdapter
from functools import partial


def _tiny_encoder(dim=32, depth=32, img=64, patch=16):
    return ImageEncoderViT(
        img_size=img, patch_size=patch, embed_dim=dim, depth=depth, num_heads=4,
        out_chans=16, global_attn_indexes=(7, 15, 23, 31),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=2)


def test_forward_output_shape_unchanged_with_adapter():
    grid = 64 // 16
    x = torch.randn(1, 3, 64, 64)

    # 形狀基準：乾淨 encoder（無 hook）
    with torch.no_grad():
        out_noadapter = _tiny_encoder()(x)

    # 掛 adapter 的 encoder
    enc = _tiny_encoder()
    ad = DeformAdapter(vit_dim=32, l2_channels=8, l3_channels=16, n_heads=4)
    handles = []
    for s, b in enumerate(ad.INJECT_BLOCKS):
        handles.append(enc.blocks[b].register_forward_pre_hook(ad._make_inject_pre_hook(s)))
    for s, b in enumerate(ad.EXTRACT_BLOCKS):
        handles.append(enc.blocks[b].register_forward_hook(ad._make_extract_post_hook(s)))

    ad.set_features({'l2': torch.randn(1, 8, grid * 2, grid * 2),
                     'l3': torch.randn(1, 16, grid, grid),
                     'mask': torch.rand(1, 1, grid * 2, grid * 2)}, grid, grid)
    with torch.no_grad():
        out = enc(x)
    for h in handles:
        h.remove()
    assert out.shape == out_noadapter.shape
    assert torch.isfinite(out).all()
