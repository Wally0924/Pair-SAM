# Cross-Attention Injector v5 Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 Q=ViT.detach()、全維 1280、Xavier init 的 Cross-Attention 取代 v4 SAM-Adapter MLP，解決梯度死鎖並降低 inject_cos_sim，目標 val_mIoU ≥ 61%。

**Architecture:** Q 來自 ViT token 的 stop_gradient copy（場景感知但無梯度耦合），K/V 來自 VGG l2+l3 池化後的線性投影（768→256），MHA(embed_dim=1280, kdim=256, vdim=256, heads=4)，所有投影 Xavier init，以 gate warmup 取代 zero-init 保護穩定性。

**Tech Stack:** PyTorch 2.x（`nn.MultiheadAttention` batch_first=True）、Python 3.10、conda env `sam_env`

---

## File Map

| 動作 | 檔案 |
|------|------|
| 覆寫 | `segment-anything/tests/test_vgg_adapter_pre_hook.py` |
| 覆寫 | `segment-anything/segment_anything/modeling/vgg_adapter.py` |
| 修改 | `segment-anything/segment_anything/modeling/weather_sam.py:81-84` |

`weather_trainer.py` 和 `train.py` 不需修改（diagnostic 屬性名稱與現有 trainer 完全相容）。

---

## Task 1：以 TDD 替換 Adapter 測試（v5 API）

**Files:**
- Overwrite: `segment-anything/tests/test_vgg_adapter_pre_hook.py`

- [ ] **Step 1：覆寫測試檔**

```python
# segment-anything/tests/test_vgg_adapter_pre_hook.py
"""
測試 MultiScaleCrossAttnInjector v5（Cross-Attention，Q=ViT.detach()，全維，Xavier init）
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
        vit_dim=64, l2_channels=16, l3_channels=32,
        d_attn=32, pool_size=4, num_heads=4,
    )


def test_no_mlp_modules():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'vgg_mlp_downs'), "v4 MLP module must not exist in v5"
    assert not hasattr(inj, 'vgg_mlp_ups'),   "v4 MLP module must not exist in v5"


def test_has_cross_attn_modules():
    inj = MultiScaleCrossAttnInjector()
    assert hasattr(inj, 'k_projs')     and len(inj.k_projs)     == 4
    assert hasattr(inj, 'v_projs')     and len(inj.v_projs)     == 4
    assert hasattr(inj, 'cross_attns') and len(inj.cross_attns) == 4


def test_no_q_bottleneck():
    inj = MultiScaleCrossAttnInjector()
    assert not hasattr(inj, 'q_down_projs'), "Q must not have bottleneck projection in v5"


def test_gate_initial_value_approx_0_05():
    inj = MultiScaleCrossAttnInjector()
    gate = F.softplus(inj.gates[0])
    assert abs(gate.item() - 0.05) < 0.005, f"expected ≈0.05, got {gate.item():.4f}"


def test_xavier_init_not_zero():
    inj = MultiScaleCrossAttnInjector()
    for i in range(4):
        assert inj.k_projs[i].weight.abs().max().item() > 0.0, \
            f"k_projs[{i}] must be Xavier-init, not zero"
        assert inj.v_projs[i].weight.abs().max().item() > 0.0, \
            f"v_projs[{i}] must be Xavier-init, not zero"


def test_inject_shape_preserved():
    inj = _small()
    B, H, W, C = 2, 8, 8, 64
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(torch.randn(B, H, W, C), 0)
    assert out.shape == (B, H, W, C)


def test_delta_driven_by_vgg_not_vit():
    """固定 Q（ViT token），改變 VGG K/V → output 應改變。"""
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

    assert (out_a - out_b).abs().max().item() > 0.01, \
        "Different VGG feats must produce different output"


def test_vit_q_detached_no_grad():
    """Q 必須 detach：梯度只走殘差路徑，vit_input.grad 應為全 1。
    原理：out = q + gate*delta；若 Q 正確 detach，delta 對 vit_input 無梯度，
    ∂sum(out)/∂vit_input = 1（all-ones）。若未 detach 則 grad ≠ ones。
    """
    inj = _small()
    B, H, W, C = 1, 4, 4, 64
    vit_input = torch.randn(B, H, W, C, requires_grad=True)
    inj.set_features({'l2': torch.randn(B, 16, H, W), 'l3': torch.randn(B, 32, H, W)})
    out = inj._inject_at_stage(vit_input, 0)
    out.sum().backward()
    assert vit_input.grad is not None
    assert torch.allclose(vit_input.grad, torch.ones_like(vit_input)), \
        "grad must be all-ones (residual only); non-ones means Q is NOT detached"


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

- [ ] **Step 2：確認測試全部失敗（v4 API 不符合 v5 合約）**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | tail -20
```

預期：多數 FAILED（`vgg_mlp_downs` 存在、`cross_attns` 不存在等）

- [ ] **Step 3：Commit**

```bash
git add segment-anything/tests/test_vgg_adapter_pre_hook.py
git commit -m "test: replace adapter tests for v5 cross-attn design"
```

---

## Task 2：重寫 vgg_adapter.py（v5 Cross-Attention）

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
    Multi-scale VGG Feature Injector（Cross-Attention v5）。

    設計：
      - Q = ViT token（.detach()，stop_gradient）— 場景感知但梯度不回傳至 ViT
      - K, V = VGG feats（l2+l3 concat → pool(pool_size²) → Linear(kv_in, d_attn)）
      - MHA(embed_dim=vit_dim, kdim=d_attn, vdim=d_attn, num_heads, batch_first=True)
      - 所有投影 Xavier init（PyTorch MHA/Linear 預設），無 zero-init 梯度死鎖
      - Gate: softplus(init≈0.05) + trainer gate warmup 保護穩定性

    相對 v4（MLP）的改動：
      - 移除 vgg_mlp_downs / vgg_mlp_ups（MLP 路徑）
      - 新增 k_projs / v_projs / cross_attns（Cross-Attention 路徑）
      - Q 不經瓶頸壓縮，保留完整 vit_dim 語意

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

    Diagnostics（trainer 相容）：
        _last_inject_cos_sim  : float — cos(q, injected) 4 stage 均值
        _last_gate_val        : float — softplus(gate) 4 stage 均值
        _last_delta_norm_ratio: float — ||gate*delta|| / ||q||
        _stage_cos_sims       : list[float] — per-stage cos_sim
        _stage_gate_vals      : list[float] — per-stage gate 值
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]

    def __init__(
        self,
        vit_dim: int = 1280,
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_attn: int = 256,
        pool_size: int = 32,
        num_heads: int = 4,
        gate_init: float = _DEFAULT_GATE_INIT,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.pool_size = pool_size
        kv_in = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # K/V projections：VGG concat → d_attn（Xavier init by default）
        self.k_projs = nn.ModuleList([
            nn.Linear(kv_in, d_attn) for _ in range(num_stages)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(kv_in, d_attn) for _ in range(num_stages)
        ])

        # Cross-attention：Q(vit_dim) × K(d_attn) × V(d_attn) → delta(vit_dim)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=vit_dim,
                kdim=d_attn,
                vdim=d_attn,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_stages)
        ])

        # Gate：softplus(raw_gate)，初始 ≈ 0.05，無上界
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
        q = output.reshape(B, H * W, C)   # 有梯度，用於殘差加法
        Q = q.detach()                     # 無梯度，用於 attention key selection

        # VGG feats → interpolate to (H,W) → concat → pool → flatten
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)  # (B, kv_in, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, self.pool_size ** 2, -1)  # (B, P², kv_in)

        # K, V projections（Xavier init）
        K = self.k_projs[stage_idx](f_flat)   # (B, P², d_attn)
        V = self.v_projs[stage_idx](f_flat)   # (B, P², d_attn)

        # Cross-attention：Q(detach) × K × V → delta
        delta, _ = self.cross_attns[stage_idx](Q, K, V)  # (B, H*W, vit_dim)

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

- [ ] **Step 2：執行測試，確認全部通過**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/test_vgg_adapter_pre_hook.py -v 2>&1 | tail -20
```

預期：11 passed

- [ ] **Step 3：Commit**

```bash
git add segment-anything/segment_anything/modeling/vgg_adapter.py
git commit -m "feat: replace MLP with cross-attn v5 (Q=ViT.detach, full-dim, Xavier init)"
```

---

## Task 3：更新 weather_sam.py 的 Injector 初始化

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py:81-84`

- [ ] **Step 1：修改 vgg_injector 初始化**

找到（約第 81–84 行）：

```python
# Before
self.vgg_injector = MultiScaleCrossAttnInjector(
    vit_dim=_vit_dim, l2_channels=256, l3_channels=512,
    d_hidden=256, pool_size=32,
)
```

改為：

```python
# After
self.vgg_injector = MultiScaleCrossAttnInjector(
    vit_dim=_vit_dim, l2_channels=256, l3_channels=512,
    d_attn=256, pool_size=32, num_heads=4,
)
```

- [ ] **Step 2：確認 WeatherSAM 可正常 import 與初始化**

```bash
conda run -n sam_env python -c "
import torch.nn.functional as F
from segment_anything.build_weather_sam import build_weather_sam_vit_b
m = build_weather_sam_vit_b()
m.enable_vgg_adapter('pre')
print('inject_blocks:', m.vgg_injector.INJECT_BLOCKS)
print('cross_attns count:', len(m.vgg_injector.cross_attns))
print('gate_init:', round(float(F.softplus(m.vgg_injector.gates[0])), 4))
print('has k_projs:', hasattr(m.vgg_injector, 'k_projs'))
print('has vgg_mlp_downs:', hasattr(m.vgg_injector, 'vgg_mlp_downs'))
"
```

預期輸出：
```
inject_blocks: [7, 15, 23, 31]
cross_attns count: 4
gate_init: 0.05
has k_projs: True
has vgg_mlp_downs: False
```

- [ ] **Step 3：Commit**

```bash
git add segment-anything/segment_anything/modeling/weather_sam.py
git commit -m "feat: update vgg_injector init to v5 API (cross-attn, d_attn=256, num_heads=4)"
```

---

## Task 4：整合驗證

- [ ] **Step 1：完整測試套件**

```bash
conda run -n sam_env python -m pytest segment-anything/tests/ -v 2>&1 | tail -30
```

預期：全部 PASSED，無 ImportError

- [ ] **Step 2：Forward pass smoke test（確認梯度路徑正確）**

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
    inj = model.vgg_injector
    print('output masks shape:', out[0]['masks'].shape)
    print('gate val:         ', round(inj._last_gate_val, 5))
    print('inject_cos_sim:   ', round(inj._last_inject_cos_sim, 5))
    print('delta_norm_ratio: ', round(inj._last_delta_norm_ratio, 5))
    print('cross_attns[0] type:', type(inj.cross_attns[0]).__name__)
" 2>&1
```

預期：
- `gate val ≈ 0.05`
- `inject_cos_sim` 為合理浮點數（非 NaN，可能 < 1.0，因 delta 非零）
- `delta_norm_ratio > 0`（Xavier init 確保 delta 非零）
- `cross_attns[0] type: MultiheadAttention`

- [ ] **Step 3：若有未 commit 的變更，最終 commit**

```bash
git status
# 若有未提交的修改：
git add -A
git commit -m "chore: integration check for cross-attn v5 adapter"
```
