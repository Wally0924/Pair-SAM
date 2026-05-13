# Pre-Hook Adapter Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 `MultiScaleCrossAttnInjector` 從 post-block 注入（`register_forward_hook`）改為 pre-block 注入（`register_forward_pre_hook`），讓晴天補償信號在 ViT-H Block 7/15/23/31 的 global self-attention **執行前**進入 token，使補償信號直接參與 attention Q/K/V 計算，對齊 SAM-Adapter 論文的設計精神。

**Architecture:** 在 `vgg_adapter.py` 新增 `_make_pre_hook()` 方法（與現有 `_make_hook()` 並存，供 ablation 比較），內部呼叫不變的 `_inject_at_stage()`；在 `weather_sam.py` 將 `enable_vgg_adapter()` 改為使用 `register_forward_pre_hook`，並更新 docstring 與 log 訊息。`_inject_at_stage()` 的張量形狀 `(B, H, W, C)` 在 pre/post hook 中完全相同，無需修改。

**Tech Stack:** Python 3.x, PyTorch `register_forward_pre_hook`, ViT-H encoder (`image_encoder.blocks`), `MultiScaleCrossAttnInjector`

---

## 檔案異動地圖

| 檔案 | 異動類型 | 說明 |
|------|----------|------|
| `segment-anything/segment_anything/modeling/vgg_adapter.py` | Modify (line 115–119) | 新增 `_make_pre_hook()` 方法，保留 `_make_hook()` 供 ablation |
| `segment-anything/segment_anything/modeling/weather_sam.py` | Modify (line 93–120) | `enable_vgg_adapter()` 改用 pre-hook；更新 docstring 與 print |
| `segment-anything/tests/test_vgg_adapter_pre_hook.py` | Create | 驗證 pre-hook 行為正確性的獨立測試腳本 |

---

## 背景知識（必讀）

### Post-hook vs Pre-hook 的差異

```
Post-hook（現有）：                    Pre-hook（目標）：
Block i self-attn（先執行）            Hook 攔截 Block i 輸入
        │                                      │
        ▼ hook 攔截輸出                         ▼
Cross-Attn with VGG                   Cross-Attn with VGG
        │                                      │
        ▼                                      ▼ modified input
修改後輸出 → Block i+1                  Block i self-attn（後執行，使用修改後輸入）
                                               │
                                               ▼
                                       Block i+1
```

### PyTorch pre-hook API

```python
# register_forward_hook: hook(module, input, output) → return value replaces output
# register_forward_pre_hook: hook(module, input) → return value replaces input
#   input 是 tuple，必須回傳 tuple 或 None（None 表示不修改）

handle = block.register_forward_pre_hook(lambda module, input: (modified_x,))
```

### `_inject_at_stage` 不需改動

`_inject_at_stage(self, output, stage_idx)` 的參數名雖叫 `output`，但邏輯只看張量的形狀 `(B, H, W, C)`。Pre-hook 時傳入的是 block 的 **input**（同樣是 `(B, H, W, C)`），函式行為完全相同。

---

## Task 1：建立測試腳本（驗證 pre-hook 行為）

**Files:**
- Create: `segment-anything/tests/test_vgg_adapter_pre_hook.py`

- [ ] **Step 1：撰寫測試腳本**

建立 `segment-anything/tests/test_vgg_adapter_pre_hook.py`，內容如下：

```python
"""
測試 MultiScaleCrossAttnInjector 的 pre-hook 行為。
執行環境：conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v
"""
import inspect
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
    # gate ≈ 0.007（非零），delta 不全為零 → block 收到的 input 與原始 x 應有差異
    # 注意：gate 極小，差異也極小但不應為零
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

    import math
    assert not math.isnan(injector._last_inject_cos_sim), "_last_inject_cos_sim is NaN"
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
```

- [ ] **Step 2：確認測試目前失敗（因 `_make_pre_hook` 尚未實作）**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | head -40
```

預期輸出：`FAILED test_make_pre_hook_exists` / `AttributeError: 'MultiScaleCrossAttnInjector' object has no attribute '_make_pre_hook'`

---

## Task 2：在 `vgg_adapter.py` 實作 `_make_pre_hook`

**Files:**
- Modify: `segment-anything/segment_anything/modeling/vgg_adapter.py:115-119`

- [ ] **Step 1：在 `_make_hook` 之後新增 `_make_pre_hook` 方法**

在 [vgg_adapter.py](segment-anything/segment_anything/modeling/vgg_adapter.py) 第 119 行（`_make_hook` 結尾的空行）之後，插入以下方法：

```python
    def _make_pre_hook(self, stage_idx: int):
        """為指定 stage 建立 forward pre-hook closure。

        Pre-hook 在 Block 執行前攔截輸入，補償 delta 會直接參與
        Block 自身的 global self-attention Q/K/V 計算（對齊 SAM-Adapter 設計）。

        PyTorch pre-hook 規範：回傳 (modified_input,) tuple；
        若回傳 None 則輸入不被修改。
        """
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook
```

修改後 `vgg_adapter.py` 第 115–124 行應如下：

```python
    def _make_hook(self, stage_idx: int):
        """為指定 stage 建立 forward hook closure，正確捕捉 stage_idx。"""
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    def _make_pre_hook(self, stage_idx: int):
        """為指定 stage 建立 forward pre-hook closure。

        Pre-hook 在 Block 執行前攔截輸入，補償 delta 會直接參與
        Block 自身的 global self-attention Q/K/V 計算（對齊 SAM-Adapter 設計）。

        PyTorch pre-hook 規範：回傳 (modified_input,) tuple；
        若回傳 None 則輸入不被修改。
        """
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook
```

- [ ] **Step 2：確認語法正確**

```bash
conda run -n sam_env python -c "
import ast, sys
with open('segment-anything/segment_anything/modeling/vgg_adapter.py') as f:
    src = f.read()
ast.parse(src)
print('AST OK')
"
```

預期輸出：`AST OK`

- [ ] **Step 3：執行測試，確認 Task 1 的測試通過**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | tail -20
```

預期輸出：所有 5 個測試 `PASSED`

---

## Task 3：更新 `enable_vgg_adapter` 使用 pre-hook

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py:93-120`

- [ ] **Step 1：替換 `enable_vgg_adapter` 方法主體**

將 [weather_sam.py](segment-anything/segment_anything/modeling/weather_sam.py) 第 93–120 行整個 `enable_vgg_adapter` 方法替換如下：

```python
    def enable_vgg_adapter(self, mode: str = 'pre'):
        """啟用 MultiScaleCrossAttnInjector，在 ViT-H Block [7, 15, 23, 31] 各注冊一個 hook。

        Args:
            mode: 'pre'（預設）使用 register_forward_pre_hook，補償信號參與 block 自身
                  global self-attention（對齊 SAM-Adapter 設計）。
                  'post' 使用 register_forward_hook（原有行為，供 ablation 比較）。

        4 個注入點對應 ViT-H 的 global attention blocks（window_size=0），
        使 encoder early/mid/late 各段都能感知對齊後的晴天參考特徵。
        每次呼叫前先移除舊 hook 避免重複注入。
        """
        for handle in self._adapter_hook_handles:
            handle.remove()
        self._adapter_hook_handles = []

        all_inject_blocks = self.vgg_injector.INJECT_BLOCKS
        n_blocks = len(self.image_encoder.blocks)
        inject_blocks = [b for b in all_inject_blocks if b < n_blocks]
        if len(inject_blocks) != len(all_inject_blocks):
            warnings.warn(
                f"[WeatherSAM] Some INJECT_BLOCKS out of range for {n_blocks}-block encoder; "
                f"using {inject_blocks} instead of {all_inject_blocks}.",
                stacklevel=2,
            )

        for stage_idx, block_idx in enumerate(inject_blocks):
            target_block = self.image_encoder.blocks[block_idx]
            if mode == 'pre':
                handle = target_block.register_forward_pre_hook(
                    self.vgg_injector._make_pre_hook(stage_idx)
                )
            else:
                handle = target_block.register_forward_hook(
                    self.vgg_injector._make_hook(stage_idx)
                )
            self._adapter_hook_handles.append(handle)

        self.use_vgg_adapter = True
        print(f'[WeatherSAM] WarpedVGG Adapter ({mode}-hook) enabled at ViT Blocks {inject_blocks}.')
```

- [ ] **Step 2：確認語法正確**

```bash
conda run -n sam_env python -c "
import ast
with open('segment-anything/segment_anything/modeling/weather_sam.py') as f:
    src = f.read()
ast.parse(src)
print('AST OK')
"
```

預期輸出：`AST OK`

- [ ] **Step 3：確認 `enable_vgg_adapter()` 預設呼叫（無參數）仍走 pre-hook**

```bash
conda run -n sam_env python -c "
import inspect, sys
sys.path.insert(0, 'segment-anything')
from segment_anything.modeling.weather_sam import WeatherSAM
src = inspect.getsource(WeatherSAM.enable_vgg_adapter)
assert \"mode: str = 'pre'\" in src, 'default mode must be pre'
assert 'register_forward_pre_hook' in src, 'pre-hook registration missing'
assert 'register_forward_hook' in src, 'post-hook fallback missing (needed for ablation)'
print('API check OK')
"
```

預期輸出：`API check OK`

---

## Task 4：Smoke Test — 完整模型 forward pass 驗證

**Files:**
- No file changes; this is a verification step only.

- [ ] **Step 1：執行 smoke test，確認 pre-hook 模式下 forward pass 正常**

```bash
conda run -n sam_env python -c "
import torch, sys, warnings
sys.path.insert(0, 'segment-anything')
warnings.filterwarnings('ignore')
from segment_anything.build_weather_sam import build_weather_sam_vit_h

# 建立 CPU 版本（無需 checkpoint，驗證架構邏輯）
model = build_weather_sam_vit_h(checkpoint=None)
model.eval()

# 啟用 pre-hook（預設）
model.enable_vgg_adapter()
assert model.use_vgg_adapter, 'use_vgg_adapter should be True'
assert len(model._adapter_hook_handles) == 4, f'expected 4 hooks, got {len(model._adapter_hook_handles)}'

# 準備最小輸入
B = 1
img = torch.zeros(B, 3, 1024, 1024)
clear = torch.zeros(B, 3, 1024, 1024)
gt = torch.zeros(B, 1024, 1024, dtype=torch.long)

batched_input = [{
    'image': img[0],
    'clear_image': clear[0],
    'original_size': (1024, 1024),
    'text_prompts': ['road', 'sky'],
    'condition_id': torch.tensor(0),
}]

with torch.no_grad():
    outputs = model(batched_input)

assert len(outputs) == 1, f'expected 1 output, got {len(outputs)}'
assert 'masks' in outputs[0], 'masks key missing in output'
assert outputs[0]['masks'].shape[-2:] == (1024, 1024), \
    f'mask shape wrong: {outputs[0][\"masks\"].shape}'

# 確認 diagnostics 被更新
inj = model.vgg_injector
assert inj._last_gate_val > 0.0, '_last_gate_val not updated'
assert inj._last_inject_cos_sim <= 1.0, '_last_inject_cos_sim out of range'

print('=== Smoke Test PASSED ===')
print(f'  mask shape    : {outputs[0][\"masks\"].shape}')
print(f'  gate_val      : {inj._last_gate_val:.6f}  (expect ~0.0067)')
print(f'  cos_sim       : {inj._last_inject_cos_sim:.6f}')
print(f'  delta_norm_ratio: {inj._last_delta_norm_ratio:.6f}')
print(f'  hook handles  : {len(model._adapter_hook_handles)}')
" 2>&1
```

預期輸出：
```
=== Smoke Test PASSED ===
  mask shape    : torch.Size([1, 2, 1024, 1024])
  gate_val      : 0.006693  (expect ~0.0067)
  cos_sim       : ~1.000000
  delta_norm_ratio: <0.1
  hook handles  : 4
```

- [ ] **Step 2：確認 post-hook ablation 模式仍可正常運作**

```bash
conda run -n sam_env python -c "
import torch, sys, warnings
sys.path.insert(0, 'segment-anything')
warnings.filterwarnings('ignore')
from segment_anything.build_weather_sam import build_weather_sam_vit_h

model = build_weather_sam_vit_h(checkpoint=None)
model.eval()

# 用 post-hook（ablation 模式）
model.enable_vgg_adapter(mode='post')
assert len(model._adapter_hook_handles) == 4

batched_input = [{
    'image': torch.zeros(3, 1024, 1024),
    'clear_image': torch.zeros(3, 1024, 1024),
    'original_size': (1024, 1024),
    'text_prompts': ['road'],
    'condition_id': torch.tensor(0),
}]

with torch.no_grad():
    outputs = model(batched_input)

assert 'masks' in outputs[0]
print('=== Ablation post-hook mode PASSED ===')
" 2>&1
```

預期輸出：`=== Ablation post-hook mode PASSED ===`

---

## Task 5：更新 `train.py` 的呼叫方式（可選，保持一致性）

**Files:**
- Modify: `segment-anything/train.py`（找到 `enable_vgg_adapter` 的呼叫處）

- [ ] **Step 1：確認 train.py 呼叫方式**

```bash
grep -n "enable_vgg_adapter" segment-anything/train.py
```

- [ ] **Step 2：若呼叫為 `model.enable_vgg_adapter()` 無參數，維持不動（預設即 pre-hook）**

無需修改。若有顯式參數，確認為 `mode='pre'`。

---

## Task 6：Commit

- [ ] **Step 1：確認所有測試通過**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v
```

預期：5/5 PASSED

- [ ] **Step 2：Commit**

```bash
git add segment-anything/segment_anything/modeling/vgg_adapter.py \
        segment-anything/segment_anything/modeling/weather_sam.py \
        segment-anything/tests/test_vgg_adapter_pre_hook.py \
        docs/superpowers/plans/2026-05-12-pre-hook-adapter-migration.md

git commit -m "$(cat <<'EOF'
feat: migrate VGG adapter from post-hook to pre-hook injection

Switch MultiScaleCrossAttnInjector from register_forward_hook to
register_forward_pre_hook at ViT-H Blocks [7,15,23,31], aligning with
SAM-Adapter's design so weather compensation signals participate in
each global block's self-attention Q/K/V computation.

enable_vgg_adapter(mode='pre'|'post') keeps post-hook accessible for
ablation comparison. No parameter changes; _inject_at_stage() unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `_make_pre_hook` 新增（Task 2）
- ✅ `_make_hook` 保留供 ablation（Task 2 — 不刪除）
- ✅ `enable_vgg_adapter` 預設改 pre-hook，可選 post（Task 3）
- ✅ API 向後相容：無參數呼叫仍有效（Task 3 default arg）
- ✅ 測試驗證 signature / 回傳格式 / 實際修改輸入 / diagnostics 更新（Task 1）
- ✅ Smoke test 含 pre 和 post 兩種模式（Task 4）

**Placeholder scan:** 無 TBD / TODO。

**Type consistency:** `_make_pre_hook(stage_idx: int)` / `_make_hook(stage_idx: int)` 簽名一致；`enable_vgg_adapter(mode: str = 'pre')` 新參數有預設值，向後相容。
