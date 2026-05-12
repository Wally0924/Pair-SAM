"""
測試 MultiScaleCrossAttnInjector 的 pre-hook 行為。
執行環境：conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v
"""
import inspect
import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _make_injector_with_feats(batch_size: int = 1) -> MultiScaleCrossAttnInjector:
    """建立已設定 multi_scale_feats 的 injector（供各測試重用）。"""
    injector = MultiScaleCrossAttnInjector(
        vit_dim=1280, d_attn=256, l2_channels=256, l3_channels=512,
        d_kv=64, pool_size=32, num_heads=4, gate_init=-5.0,
    )
    injector.set_features({
        'l2': torch.zeros(batch_size, 256, 64, 64),
        'l3': torch.zeros(batch_size, 512, 64, 64),
    })
    return injector


# ── Test 1：_make_pre_hook 存在且 signature 正確 ──────────────────────────────

def test_make_pre_hook_exists():
    injector = MultiScaleCrossAttnInjector()
    assert hasattr(injector, '_make_pre_hook'), \
        "_make_pre_hook method missing from MultiScaleCrossAttnInjector"


def test_pre_hook_takes_two_args():
    """pre-hook 必須接受 (module, input) 共 2 個參數，而非 post-hook 的 3 個。"""
    injector = MultiScaleCrossAttnInjector()
    hook_fn = injector._make_pre_hook(0)
    sig = inspect.signature(hook_fn)
    n_params = len(sig.parameters)
    assert n_params == 2, (
        f"_make_pre_hook 的 closure 必須接受 2 個參數 (module, input)，實際得到 {n_params}"
    )


# ── Test 2：pre-hook 回傳值格式正確 ──────────────────────────────────────────

def test_pre_hook_returns_tuple():
    """PyTorch pre-hook 必須回傳 tuple 或 None；這裡要求回傳修改後的 tuple。"""
    injector = _make_injector_with_feats()
    hook_fn = injector._make_pre_hook(0)

    class FakeBlock(nn.Module):
        def forward(self, x): return x

    x = torch.zeros(1, 64, 64, 1280)
    result = hook_fn(FakeBlock(), (x,))

    assert isinstance(result, tuple), \
        f"_make_pre_hook 必須回傳 tuple，得到 {type(result)}"
    assert len(result) == 1, \
        f"回傳 tuple 長度必須為 1，得到 {len(result)}"
    assert result[0].shape == x.shape, \
        f"輸出形狀 {result[0].shape} 必須等於輸入形狀 {x.shape}"


# ── Test 3：pre-hook 實際改變 Block 的輸入（注入在 forward 之前）────────────────

def test_pre_hook_modifies_block_input():
    """
    驗證 pre-hook 在 block.forward() 執行前修改了輸入。
    做法：block.forward() 記錄自己收到的 input；
    若有 pre-hook，block 看到的 input 應與原始 x 不同（gate*delta != 0）。
    """
    injector = _make_injector_with_feats()

    received_inputs = []

    class RecordingBlock(nn.Module):
        def forward(self, x):
            received_inputs.append(x.detach().clone())
            return x

    block = RecordingBlock()
    block.register_forward_pre_hook(injector._make_pre_hook(0))

    x_original = torch.randn(1, 64, 64, 1280)
    _ = block(x_original)

    assert len(received_inputs) == 1
    # gate ≈ 0.007（非零），delta 不全為零 → block 收到的 input 應與原始 x 應有差異
    diff = (received_inputs[0] - x_original).abs().max().item()
    assert diff > 0.0, (
        f"pre-hook 應修改 block 輸入（gate*delta != 0），但 max diff = {diff}"
    )


# ── Test 4：_stages_fired 與 diagnostics 在 pre-hook 模式仍正確更新 ─────────

def test_diagnostics_updated_after_four_stages():
    """
    4 個 stage 的 pre-hook 全部觸發後，
    _last_inject_cos_sim / _last_gate_val / _last_delta_norm_ratio 必須被更新。
    """
    injector = MultiScaleCrossAttnInjector(gate_init=-5.0)
    injector.set_features({
        'l2': torch.randn(1, 256, 64, 64),
        'l3': torch.randn(1, 512, 64, 64),
    })

    class FakeBlock(nn.Module):
        def forward(self, x): return x

    hooks = []
    for stage_idx in range(4):
        blk = FakeBlock()
        handle = blk.register_forward_pre_hook(injector._make_pre_hook(stage_idx))
        hooks.append((blk, handle))

    x = torch.randn(1, 64, 64, 1280)
    for blk, _ in hooks:
        blk(x)

    assert not math.isnan(injector._last_inject_cos_sim), "_last_inject_cos_sim is NaN"
    # 容差：sigmoid(-5)≈0.00669，0.02 為 3× 安全邊際，覆蓋浮點誤差與數值擾動
    assert 0.0 < injector._last_gate_val < 0.02, \
        f"_last_gate_val={injector._last_gate_val:.4f} 應接近 sigmoid(-5)≈0.007"
    assert injector._last_delta_norm_ratio >= 0.0, \
        "_last_delta_norm_ratio 不應為負"


# ── Test 5：_make_hook（post-hook）仍然可用（backward compatibility）─────────

def test_post_hook_still_works():
    """保留 _make_hook 確保 ablation 實驗可切換回 post-hook。"""
    injector = _make_injector_with_feats()
    assert hasattr(injector, '_make_hook'), \
        "_make_hook 被刪除；必須保留供 ablation 使用"

    hook_fn = injector._make_hook(0)
    sig = inspect.signature(hook_fn)
    n_params = len(sig.parameters)
    assert n_params == 3, (
        f"_make_hook 的 closure 必須接受 3 個參數 (module, input, output)，得到 {n_params}"
    )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
