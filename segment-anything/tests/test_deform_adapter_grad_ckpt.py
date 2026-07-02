"""Gradient checkpointing × DeformAdapter hook 的梯度等價性。

checkpointing 在 backward 重放 block forward 時，hook 會再次觸發；
inject hook 讀取的 self._c 必須與原 forward 當下一致，否則重算的
activation 偏離原 forward，梯度悄悄算錯。本測試比對開/關 checkpointing
的參數梯度，要求完全一致（同輸入、同權重、gamma 設為非零使注入生效）。
"""
import torch, torch.nn as nn, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.deform_adapter import DeformAdapter


def _build(seed=0, dim=32, depth=32, img=64, patch=16):
    torch.manual_seed(seed)
    enc = ImageEncoderViT(
        img_size=img, patch_size=patch, embed_dim=dim, depth=depth, num_heads=4,
        out_chans=16, global_attn_indexes=(7, 15, 23, 31),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=2)
    ad = DeformAdapter(vit_dim=dim, l2_channels=8, l3_channels=16, l4_channels=16,
                       n_heads=4)
    with torch.no_grad():
        for inj in ad.injectors:
            inj.gamma.fill_(0.5)          # 注入生效，讓 c 路徑有實質梯度
    for s, b in enumerate(ad.INJECT_BLOCKS):
        enc.blocks[b].register_forward_pre_hook(ad._make_inject_pre_hook(s))
    for s, b in enumerate(ad.EXTRACT_BLOCKS):
        enc.blocks[b].register_forward_hook(ad._make_extract_post_hook(s))
    return enc, ad


def _grads(use_ckpt):
    grid = 64 // 16
    enc, ad = _build()
    enc.image_encoder_dummy = None
    enc.use_checkpoint = use_ckpt
    enc.train()
    torch.manual_seed(1)
    feats = {'l2': torch.randn(1, 8, grid * 2, grid * 2),
             'l3': torch.randn(1, 16, grid, grid),
             'l4': torch.randn(1, 16, grid // 2, grid // 2),
             'mask': torch.rand(1, 1, grid * 2, grid * 2)}
    x = torch.randn(1, 3, 64, 64)
    ad.set_features(feats, grid, grid)
    out = enc(x)
    out.square().mean().backward()
    return {n: p.grad.clone() for n, p in ad.named_parameters()
            if p.grad is not None}


def test_ckpt_grads_match_nockpt():
    """快照重放正確時，重算 forward 與原 forward 逐位元一致 → 梯度以「相對」
    標準比對（toy loss 梯度僅 1e-7 量級，絕對容差會掩蓋 100%+ 的相對誤差）。"""
    g0 = _grads(use_ckpt=False)
    g1 = _grads(use_ckpt=True)
    assert g0.keys() == g1.keys(), (
        f"梯度參數集合不同：only-nockpt={g0.keys()-g1.keys()}, only-ckpt={g1.keys()-g0.keys()}")
    bad = []
    for n in g0:
        d = (g0[n] - g1[n]).abs().max().item()
        ref = max(g0[n].abs().max().item(), 1e-30)
        if d / ref > 1e-5:
            bad.append(f"{n}: rel={d/ref:.3e} max|Δ|={d:.3e} (ref max={ref:.3e})")
    assert not bad, "checkpointing 改變了梯度：\n" + "\n".join(bad[:10])
