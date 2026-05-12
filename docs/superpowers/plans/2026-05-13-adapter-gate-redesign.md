# Adapter Gate Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修復 VGG Adapter gate 近乎靜止（inject_gate ≈ 0.007）的問題，同時補齊 train_log 可觀測性並移除 Focal Loss 的信心壓制效果。

**Architecture:** 以 SAM-Adapter 風格 MLP 取代 Cross-Attention（解除 delta 與 ViT token 的耦合），以 softplus gate 取代 sigmoid gate（梯度強 7×），並在前 3 epoch warmup 凍結 gate 讓 main decoder 先穩定。

**Tech Stack:** PyTorch 2.x, Python 3.10, conda env `sam_env`

---

## File Map

| 動作 | 檔案 |
|------|------|
| 覆寫 | `segment-anything/segment_anything/modeling/vgg_adapter.py` |
| 覆寫 | `segment-anything/tests/test_vgg_adapter_pre_hook.py` |
| 修改 | `segment-anything/segment_anything/modeling/weather_sam.py` |
| 修改 | `segment-anything/weather_trainer.py` |
| 修改 | `segment-anything/train.py` |

---

## Task 1：以 TDD 替換 Adapter 測試

**Files:**
- Overwrite: `segment-anything/tests/test_vgg_adapter_pre_hook.py`

- [ ] **Step 1：覆寫測試檔（新 API 驗證）**

```python
# segment-anything/tests/test_vgg_adapter_pre_hook.py
"""
測試 MultiScaleCrossAttnInjector v4（SAM-Adapter 風格 MLP + softplus gate）
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector


def _small():
    return MultiScaleCrossAttnInjector(
        vit_dim=64, l2_channels=16, l3_channels=32, d_hidden=32, pool_size=4
    )


def test_no_cross_attention_modules():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'cross_attns')
    assert not hasattr(inj, 'q_down_projs')
    assert not hasattr(inj, 'q_up_projs')


def test_has_mlp_modules():
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, 'vgg_mlp_downs') and len(inj.vgg_mlp_downs) == 4
    assert hasattr(inj, 'vgg_mlp_ups')   and len(inj.vgg_mlp_ups)   == 4


def test_gate_initial_value_approx_0_05():
    inj = MultiScaleCrossAttnInjector()
    gate = F.softplus(inj.gates[0])
    assert abs(gate.item() - 0.05) < 0.005, f"expected ≈0.05, got {gate.item():.4f}"


def test_softplus_gradient_stronger_than_sigmoid():
    raw = torch.tensor(-2.9444, requires_grad=True)
    F.softplus(raw).backward()
    sp_grad = raw.grad.item()

    raw2 = torch.tensor(-5.0, requires_grad=True)
    torch.sigmoid(raw2).backward()
    sig_grad = raw2.grad.item()

    assert sp_grad > sig_grad * 5, f"softplus grad {sp_grad:.4f} should be >> sigmoid grad {sig_grad:.4f}"


def test_mlp_up_zero_initialized():
    inj = MultiScaleCrossAttnInjector()
    for i, proj in enumerate(inj.vgg_mlp_ups):
        assert proj.weight.abs().max().item() == 0.0, f"vgg_mlp_ups[{i}] not zero-init"


def test_inject_shape_preserved():
    inj = _small()
    B, H, W, C = 2, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)


def test_delta_driven_by_vgg_not_vit():
    """固定 ViT token，改變 VGG feats → output 應改變。"""
    inj = _small()
    with torch.no_grad():
        for g in inj.gates:
            g.fill_(5.0)
    B, H, W, C = 1, 8, 8, 64
    vit = torch.randn(B, H, W, C)

    inj.set_features({'l2': torch.ones(B, 16, H, W), 'l3': torch.ones(B, 32, H, W)})
    out_a = inj._inject_at_stage(vit.clone(), 0)

    inj.set_features({'l2': -torch.ones(B, 16, H, W), 'l3': -torch.ones(B, 32, H, W)})
    out_b = inj._inject_at_stage(vit.clone(), 0)

    assert (out_a - out_b).abs().max().item() > 0.01, "Different VGG feats must produce different output"


def test_diagnostics_updated_after_all_stages():
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    for i in range(4):
        inj._inject_at_stage(torch.randn(B, H, W, C), i)
    assert not math.isnan(inj._last_inject_cos_sim)
    assert inj._last_gate_val > 0.0
    assert inj._last_delta_norm_ratio >= 0.0


def test_pre_hook_returns_tuple_of_correct_shape():
    inj = _small()
    B, H, W, C = 1, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    hook = inj._make_pre_hook(0)
    result = hook(nn.Linear(1, 1), (torch.randn(B, H, W, C),))
    assert isinstance(result, tuple) and len(result) == 1
    assert result[0].shape == (B, H, W, C)


def test_make_hook_post_still_exists():
    """_make_hook (post-hook) 必須保留供 ablation 使用。"""
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, '_make_hook')
```

- [ ] **Step 2：確認測試全部失敗（新 API 尚未實作）**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | tail -20
```

預期：多數測試 FAILED（`has no attribute 'vgg_mlp_downs'` 等）

- [ ] **Step 3：Commit 測試**

```bash
git add segment-anything/tests/test_vgg_adapter_pre_hook.py
git commit -m "test: replace adapter tests for v4 MLP+softplus design"
```

---

## Task 2：重寫 vgg_adapter.py（MLP + softplus gate）

**Files:**
- Overwrite: `segment-anything/segment_anything/modeling/vgg_adapter.py`

- [ ] **Step 1：完整覆寫 vgg_adapter.py**

```python
# segment-anything/segment_anything/modeling/vgg_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # ≈ -2.9444；softplus(x) ≈ 0.05


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale VGG Feature Injector（SAM-Adapter 風格，v4）。

    設計變更（相對 v3）：
      - 移除 Cross-Attention（Q 來自 ViT token → delta 被迫與 q 相似，inject_cos_sim 高）
      - 改用 SAM-Adapter 風格 MLP：VGG feats → pool → MLP_down/GELU/MLP_up → delta
        delta 完全獨立於 ViT token，inject_cos_sim 預期從 0.79 降至 0.5 以下
      - Gate：sigmoid(-5)≈0.007 → softplus(-2.94)≈0.05，梯度強 7×，上界 ∞
      - MLP_up zero-init：初期 delta=0，與 gate warmup 雙重保護訓練穩定性

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

    Diagnostics（trainer 相容）：
        _last_inject_cos_sim  : float — 4 stage 注入前後 cosine similarity 均值
        _last_gate_val        : float — 4 stage softplus(gate) 均值
        _last_delta_norm_ratio: float — inject_delta_norm / vit_token_norm
        _stage_cos_sims       : list[float] — per-stage cos_sim
        _stage_gate_vals      : list[float] — per-stage gate 值
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]

    def __init__(
        self,
        vit_dim: int = 1280,
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_hidden: int = 256,
        pool_size: int = 32,
        gate_init: float = _DEFAULT_GATE_INIT,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.pool_size = pool_size
        kv_in = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # MLP_down: kv_in → d_hidden（per-stage）
        self.vgg_mlp_downs = nn.ModuleList([
            nn.Linear(kv_in, d_hidden) for _ in range(num_stages)
        ])

        # MLP_up: d_hidden → vit_dim（per-stage）；zero-init 避免初期擾動
        self.vgg_mlp_ups = nn.ModuleList([
            nn.Linear(d_hidden, vit_dim, bias=False) for _ in range(num_stages)
        ])
        for proj in self.vgg_mlp_ups:
            nn.init.zeros_(proj.weight)

        # Gate：softplus(raw_gate)，初始 ≈ 0.05
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            for _ in range(num_stages)
        ])

        self._multi_scale_feats: dict = None
        self._stages_fired: int = 0

        _init_gate = float(F.softplus(torch.tensor(gate_init)))
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = _init_gate
        self._last_delta_norm_ratio: float = 0.0
        self._stage_cos_sims: list = [1.0] * num_stages
        self._stage_gate_vals: list = [_init_gate] * num_stages
        self._global_step: int = 0

    def set_features(self, multi_scale_feats: dict):
        self._multi_scale_feats = multi_scale_feats
        self._stages_fired = 0

    def _make_hook(self, stage_idx: int):
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    def _make_pre_hook(self, stage_idx: int):
        def hook(module, input):
            return (self._inject_at_stage(input[0], stage_idx),)
        return hook

    def _inject_at_stage(self, output: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if self._multi_scale_feats is None:
            return output

        f_l2 = self._multi_scale_feats['l2'].to(output.device, dtype=output.dtype)
        f_l3 = self._multi_scale_feats['l3'].to(output.device, dtype=output.dtype)

        B, H, W, C = output.shape
        q = output.reshape(B, H * W, C)

        # VGG feats → interpolate to (H,W) → concat → pool → flatten
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)                          # (B, kv_in, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, self.pool_size ** 2, -1)  # (B, P², kv_in)

        # MLP: (B, P², kv_in) → (B, P², d_hidden) → (B, P², vit_dim)
        h = F.gelu(self.vgg_mlp_downs[stage_idx](f_flat))
        delta_spatial = self.vgg_mlp_ups[stage_idx](h)                     # (B, P², vit_dim)

        # Bilinear upsample: pool_size×pool_size → H×W，保留空間結構
        delta_map = delta_spatial.reshape(B, self.pool_size, self.pool_size, self.vit_dim)
        delta_map = delta_map.permute(0, 3, 1, 2)                          # (B, vit_dim, P, P)
        delta_map = F.interpolate(delta_map, size=(H, W), mode='bilinear', align_corners=False)
        delta = delta_map.permute(0, 2, 3, 1).reshape(B, H * W, self.vit_dim)  # (B, H*W, vit_dim)

        gate = F.softplus(self.gates[stage_idx])
        injected_flat = q + gate * delta
        injected = injected_flat.reshape(B, H, W, C)

        with torch.no_grad():
            cos = F.cosine_similarity(q, injected_flat, dim=-1).mean().item()
            self._stage_cos_sims[stage_idx] = cos
            self._stage_gate_vals[stage_idx] = float(gate.item())
            if stage_idx == 0:
                delta_norm = (gate * delta).norm(dim=-1).mean().item()
                vit_norm   = q.norm(dim=-1).mean().item()
                self._last_delta_norm_ratio = delta_norm / (vit_norm + 1e-8)

        self._stages_fired += 1
        if self._stages_fired >= self._num_stages:
            self._last_inject_cos_sim = float(sum(self._stage_cos_sims) / self._num_stages)
            self._last_gate_val = float(sum(self._stage_gate_vals) / self._num_stages)
            self._global_step += 1
            self._multi_scale_feats = None
            self._stages_fired = 0

        return injected
```

- [ ] **Step 2：執行測試，確認通過**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | tail -20
```

預期：全部 PASSED

- [ ] **Step 3：Commit**

```bash
git add segment-anything/segment_anything/modeling/vgg_adapter.py
git commit -m "feat: replace cross-attn with SAM-Adapter MLP, sigmoid gate with softplus"
```

---

## Task 3：更新 weather_sam.py 的 Injector 初始化

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py:80-83`

- [ ] **Step 1：修改 __init__ 中的 vgg_injector 初始化**

找到下面這段（weather_sam.py 約第 80–83 行）：

```python
# Before
self.vgg_injector = MultiScaleCrossAttnInjector(
    vit_dim=1280, d_attn=256, l2_channels=256, l3_channels=512,
    d_kv=64, pool_size=32, num_heads=4, gate_init=-5.0,
)
```

改為：

```python
# After
self.vgg_injector = MultiScaleCrossAttnInjector(
    vit_dim=1280, l2_channels=256, l3_channels=512,
    d_hidden=256, pool_size=32,
)
```

- [ ] **Step 2：確認 WeatherSAM 可正常 import 與初始化**

```bash
conda run -n sam_env python -c "
from segment_anything.build_weather_sam import build_weather_sam_vit_b
m = build_weather_sam_vit_b()
m.enable_vgg_adapter('pre')
print('OK, inject_blocks:', m.vgg_injector.INJECT_BLOCKS)
print('gate_init:', float(__import__('torch').nn.functional.softplus(m.vgg_injector.gates[0])))
"
```

預期輸出包含 `gate_init: 0.05` 附近的數字

- [ ] **Step 3：Commit**

```bash
git add segment-anything/segment_anything/modeling/weather_sam.py
git commit -m "feat: update vgg_injector init to v4 API (MLP, softplus gate)"
```

---

## Task 4：Gate Warmup（weather_trainer.py）

**Files:**
- Modify: `segment-anything/weather_trainer.py`（`__init__` 與 `train_epoch` 各一處）

- [ ] **Step 1：在 Trainer.__init__ 加入 gate params 識別**

找到 `__init__` 中 `self.mask_loss_fn = MaskLoss(...)` 這行之後（約第 111 行），加入：

```python
# Gate warmup：識別 gate 參數，前 warmup_gate_epochs 個 epoch 凍結
self.warmup_gate_epochs = getattr(args, 'warmup_gate_epochs', 3)
self._gate_params = [
    p for n, p in model.named_parameters()
    if 'vgg_injector.gates' in n
]
```

- [ ] **Step 2：在 train_epoch 開頭加入 warmup 切換**

找到 `train_epoch` 中 `self.model.train()` 這行之前，加入：

```python
# Gate warmup：前 warmup_gate_epochs 個 epoch 凍結 gate 參數
_gate_frozen = (epoch_index < self.warmup_gate_epochs)
for p in self._gate_params:
    p.requires_grad_(not _gate_frozen)
if self._gate_params:
    status = "frozen" if _gate_frozen else "trainable"
    print(f"   [Gate Warmup] epoch {epoch_index+1}: gate params {status} "
          f"(warmup ends at epoch {self.warmup_gate_epochs})")
```

- [ ] **Step 3：手動驗證 warmup 邏輯**

```bash
conda run -n sam_env python -c "
import types, torch
from segment_anything.build_weather_sam import build_weather_sam_vit_b

m = build_weather_sam_vit_b()
m.enable_vgg_adapter('pre')

gate_params = [p for n, p in m.named_parameters() if 'vgg_injector.gates' in n]
print('num gate params:', len(gate_params))

# 模擬 warmup 凍結
for p in gate_params:
    p.requires_grad_(False)
print('epoch 0 frozen:', all(not p.requires_grad for p in gate_params))

# 模擬解凍
for p in gate_params:
    p.requires_grad_(True)
print('epoch 3 trainable:', all(p.requires_grad for p in gate_params))
"
```

預期：`num gate params: 4`，兩個 bool 均為 `True`

- [ ] **Step 4：Commit**

```bash
git add segment-anything/weather_trainer.py
git commit -m "feat: add gate warmup — freeze vgg_injector.gates for first N epochs"
```

---

## Task 5：Per-stage 診斷 logging（weather_trainer.py）

**Files:**
- Modify: `segment-anything/weather_trainer.py`（train_epoch 與 validate_epoch 各兩處）

- [ ] **Step 1：在 train_epoch 的 losses dict 初始化中加入 per-stage meters**

找到 `losses` dict 定義中的 `"inject_delta_norm": AverageMeter(),` 這行之後，加入：

```python
# per-stage gate & cos（4 個 stage）
**{f"inject_gate_s{i}": AverageMeter() for i in range(4)},
**{f"inject_cos_s{i}":  AverageMeter() for i in range(4)},
```

- [ ] **Step 2：在 train_epoch 的 injector 讀取區塊加入 per-stage 更新**

找到（約第 501–504 行）：

```python
losses['inject_cos_sim'].update(float(_injector._last_inject_cos_sim), batch_size)
losses['inject_gate'].update(float(_injector._last_gate_val), batch_size)
if hasattr(_injector, '_last_delta_norm_ratio'):
    losses['inject_delta_norm'].update(float(_injector._last_delta_norm_ratio), batch_size)
```

在這段之後加入：

```python
for _si in range(_injector._num_stages):
    losses[f'inject_gate_s{_si}'].update(
        float(_injector._stage_gate_vals[_si]), batch_size)
    losses[f'inject_cos_s{_si}'].update(
        float(_injector._stage_cos_sims[_si]), batch_size)
```

- [ ] **Step 3：在 validate_epoch 的 losses dict 初始化中做同樣新增**

validate_epoch 的 `losses` dict 在約第 822–830 行，找到 `"inject_delta_norm": AverageMeter(),` 之後，加入與 Step 1 相同的兩行。

- [ ] **Step 4：在 validate_epoch 的 injector 讀取區塊加入 per-stage 更新**

找到（約第 962–965 行）的 injector 讀取區塊，在其後加入與 Step 2 相同的 for 迴圈。

- [ ] **Step 5：Commit**

```bash
git add segment-anything/weather_trainer.py
git commit -m "feat: add per-stage gate/cos AverageMeters to trainer logging"
```

---

## Task 6：Train Log CSV 欄位 + Focal/Dice 預設值（train.py）

**Files:**
- Modify: `segment-anything/train.py`

- [ ] **Step 1：在 history dict 中加入新診斷欄位**

找到（約第 396–399 行）：

```python
"train_inject_gate":     train_metrics.get("inject_gate",    0.0),
"val_inject_gate":       val_metrics.get("inject_gate",      0.0),
```

在這段之後加入：

```python
# Adapter 深度診斷
"train_inject_delta_norm": train_metrics.get("inject_delta_norm", 0.0),
"val_inject_delta_norm":   val_metrics.get("inject_delta_norm",   0.0),
"train_head_delta_norm":   train_metrics.get("head_delta_norm",   0.0),
"val_head_delta_norm":     val_metrics.get("head_delta_norm",     0.0),
# per-stage gate（s0–s3）
**{f"train_inject_gate_s{i}": train_metrics.get(f"inject_gate_s{i}", 0.0) for i in range(4)},
**{f"val_inject_gate_s{i}":   val_metrics.get(  f"inject_gate_s{i}", 0.0) for i in range(4)},
# per-stage cos_sim（s0–s3）
**{f"train_inject_cos_s{i}": train_metrics.get(f"inject_cos_s{i}", 1.0) for i in range(4)},
**{f"val_inject_cos_s{i}":   val_metrics.get(  f"inject_cos_s{i}", 1.0) for i in range(4)},
```

- [ ] **Step 2：修改 focal_weight 與 dice_weight 的 argparse 預設值**

找到（約第 188–189 行）：

```python
parser.add_argument("--focal_weight", type=float, default=5.0, help="MaskLoss (Focal) 權重")
parser.add_argument("--dice_weight",  type=float, default=0.5, help="MaskLoss (Dice) 權重")
```

改為：

```python
parser.add_argument("--focal_weight", type=float, default=0.0,
                    help="MaskLoss (Focal) 權重；0.0=停用（Dice 已足夠，Focal γ=2 會壓制信心成長）")
parser.add_argument("--dice_weight",  type=float, default=1.0,
                    help="MaskLoss (Dice) 權重；移除 Focal 後調高至 1.0")
```

- [ ] **Step 3：加入 warmup_gate_epochs argparse**

找到 `parser.add_argument("--lovasz_weight", ...)` 這行之後，加入：

```python
parser.add_argument("--warmup_gate_epochs", type=int, default=3,
                    help="前 N epoch 凍結 vgg_injector.gates，讓 main decoder 先穩定")
```

- [ ] **Step 4：確認 CSV 欄位正確產生（smoke test）**

```bash
conda run -n sam_env python -c "
import sys; sys.argv = ['train.py']
import train
import argparse
p = argparse.ArgumentParser()
train.add_arguments(p)  # 若 train.py 有 add_arguments，否則跳過
" 2>/dev/null || echo "skip (inline argparse)"

# 驗證 focal 預設值
conda run -n sam_env python -c "
import sys; sys.argv=['t','--help']
try:
    import train
except SystemExit:
    pass
" 2>&1 | grep -E "focal_weight|dice_weight|warmup_gate"
```

預期輸出包含 `focal_weight` 預設 `0.0`、`dice_weight` 預設 `1.0`、`warmup_gate_epochs` 預設 `3`

- [ ] **Step 5：Commit**

```bash
git add segment-anything/train.py
git commit -m "feat: add per-stage log cols, remove focal loss, add warmup_gate_epochs arg"
```

---

## Task 7：整合驗證

- [ ] **Step 1：執行完整測試套件**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/ -v 2>&1 | tail -30
```

預期：全部 PASSED，無 ImportError

- [ ] **Step 2：確認 WeatherSAM forward pass 不崩潰**

```bash
conda run -n sam_env python -c "
import torch
from segment_anything.build_weather_sam import build_weather_sam_vit_b

model = build_weather_sam_vit_b().cuda().eval()
model.enable_vgg_adapter('pre')

dummy_input = [{
    'image':        torch.randn(3, 1024, 1024).cuda(),
    'clear_image':  torch.randn(3, 1024, 1024).cuda(),
    'text_prompts': ['road', 'sky', 'building'],
    'original_size': (1024, 1024),
    'condition_id':  torch.tensor(1),
}]

with torch.no_grad():
    out = model(dummy_input)
    print('output masks shape:', out[0]['masks'].shape)
    print('gate val:', model.vgg_injector._last_gate_val)
    print('inject_cos_sim:', model.vgg_injector._last_inject_cos_sim)
"
```

預期：`gate val ≈ 0.05`，`inject_cos_sim` 值為合理浮點數（非 NaN）

- [ ] **Step 3：最終 Commit**

```bash
git add -A
git commit -m "chore: final integration check for adapter gate redesign (v4)"
```
