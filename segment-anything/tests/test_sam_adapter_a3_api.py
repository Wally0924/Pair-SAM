# tests/test_sam_adapter_a3_api.py
"""SameImageAdapterInjector 與 A3 hook API 的相容性(W4 基線前置)。

背景:A3 改版後 PairSAM.enable_vgg_adapter 只認 DeformAdapter 介面
(INJECT_BLOCKS/_make_inject_pre_hook + EXTRACT_BLOCKS/_make_extract_post_hook)。
本測試鎖定 SameImageAdapterInjector 必須提供同名 API,否則
--adapter_variant sam_adapter 在 enable 階段直接 AttributeError。
"""
import torch
import torch.nn as nn

from segment_anything.modeling.sam_adapter_injector import SameImageAdapterInjector


def test_a3_hook_api_present():
    inj = SameImageAdapterInjector(vit_dim=32, bottleneck=8)
    # enable_vgg_adapter 逐一取用的屬性
    assert hasattr(inj, "INJECT_BLOCKS")
    assert hasattr(inj, "EXTRACT_BLOCKS"), "缺 EXTRACT_BLOCKS(基線無 extractor,應為空 list)"
    assert inj.EXTRACT_BLOCKS == []
    assert callable(getattr(inj, "_make_inject_pre_hook", None)), \
        "缺 _make_inject_pre_hook(A3 別名,應等同 _make_pre_hook)"


def test_a3_pre_hook_injects_residual():
    """以 dummy block 註冊 A3 名稱的 pre-hook,確認注入為 (B,H,W,C) 殘差且可反傳。"""
    torch.manual_seed(0)
    inj = SameImageAdapterInjector(vit_dim=32, bottleneck=8)
    block = nn.Identity()
    block.register_forward_pre_hook(inj._make_inject_pre_hook(0))

    x = torch.randn(1, 4, 4, 32, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    # gate init≈0.05、up-proj std=0.01 → delta 微小但非零
    assert not torch.equal(out, x)
    assert torch.allclose(out, x, atol=1e-2)

    out.sum().backward()
    assert x.grad is not None
    g = inj.gates[0]
    assert g.grad is not None, "gate 未收到梯度(殘差路徑斷裂)"
