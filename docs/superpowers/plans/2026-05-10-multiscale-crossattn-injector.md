# Multi-Scale Cross-Attention VGG Injector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 VGG Adapter 從單尺度加法注入（additive injection）升級為多尺度 Cross-Attention 注入，讓 SAM ViT Encoder 各 block 能透過注意力機制自適應地從對齊後的 VGG 特徵（level2 256ch + level3 512ch）中選取補償信號，改善動態物件與細小結構的辨識。

**Architecture:**
1. `pre_align()` 改為同時回傳 VGG level2（256ch, stride8）與 level3（512ch, stride16）的扭曲特徵，使用同一 flow field；回傳 dict `{'l2': Tensor, 'l3': Tensor}`。
2. `MultiStageWarpedVGGInjector` → `MultiScaleCrossAttnInjector`：每個 stage 使用**瓶頸式 Cross-Attention**（Q 先壓縮至 d_attn=256，attention 在小維度空間計算後再投影回 1280），K/V 分開投影，取代原來的純加法 MLP adapter。
3. `WeatherSAM` 的 `set_features()` 呼叫改為傳入 dict，其餘 hook 機制不變。

**Design Fixes（v3，在 v2 基礎上修正）：**
- **移除 `out_proj` 零初始化**（v2 已修正）：直接 `q + gate * q_up(attn_out)`，gate 梯度從第一步暢通。
- **pool_size 16→32**（v2 已修正）：KV tokens 1024，保留動態物件細節。
- **K/V 分開投影**（v2 已修正）：k_proj / v_proj 各自獨立。
- **瓶頸式 Cross-Attention**（v3 新增）：Q 先壓縮到 d_attn=256 再做 attention，移除 v2 中 embed_dim=1280 的 Q projection（1.6M/stage），4 stages 總可訓練參數從 ~17M 降至 ~3.7M，適合 ACDC 1,200 張訓練集。
- **gate 初始值 -3.0 → -5.0**（v3 新增）：因 q_up_proj 使用 xavier 初始化（非零），用更保守的初始 gate ≈ 0.007 降低訓練初期的擾動量。
- **訓練初期 delta_norm 監控**（v3 新增）：前 100 steps 記錄 inject_delta_norm / vit_token_norm，若比值 > 0.1 代表注入量偏大。

**Tech Stack:** PyTorch `nn.MultiheadAttention`（`kdim/vdim` 異維 Q/KV 支援）、現有 `warp()`、`estimate_probability_of_confidence_interval()`

---

## 影響檔案總覽

| 檔案 | 動作 | 說明 |
|------|------|------|
| `segment-anything/segment_anything/modeling/fusion.py` | Modify | `pre_align()` 新增 l2 warp，回傳 dict |
| `segment-anything/segment_anything/modeling/vgg_adapter.py` | Rewrite | `MultiStageWarpedVGGInjector` → `MultiScaleCrossAttnInjector` |
| `segment-anything/segment_anything/modeling/weather_sam.py` | Modify | `set_features()` 呼叫改傳 dict；`__init__` 改用新 class |
| `segment-anything/segment_anything/modeling/__init__.py` | Modify | 更新 import（若有 export） |
| `segment-anything/segment_anything/build_weather_sam.py` | Modify | `MultiStageWarpedVGGInjector` constructor 換成 `MultiScaleCrossAttnInjector` |

---

## Task 1: 修改 `pre_align()` 回傳多尺度特徵

**Files:**
- Modify: `segment-anything/segment_anything/modeling/fusion.py:168-232`

### 設計說明

現有 `pre_align()` 的 Step 4 只取 `feats_ref[3]`（pool4, 512ch）並扭曲。
新增 Step 4b：取 `feats_ref[2]`（pool3, 256ch, stride8），用同一 `flow1` 扭曲並套用同一 `hard_mask`。
回傳由 `f_ref_warped_vgg_masked` → dict `{'l2': ..., 'l3': ...}`。

---

- [ ] **Step 1: 閱讀現有 `pre_align()` 確認 Step 4-7 位置**

讀取 `fusion.py` 第 168-232 行（已在本次 session 讀取，步驟確認）：
- Step 4（第 210-213 行）：取 `feats_ref[3]`，resize 到 `out_size`
- Step 5（第 216 行）：`warp(f_vgg_ref_l3, flow1, return_mask=True)` → `f_ref_warped_vgg`, `validity_mask`
- Step 6-7（第 219-224 行）：計算 confidence、hard_mask，產出 `f_ref_warped_vgg_masked`
- 第 232 行：`return f_ref_warped_vgg_masked`（單一 Tensor）

---

- [ ] **Step 2: 修改 `pre_align()` 加入 l2 warp 並改回傳 dict**

在 `fusion.py` 的 `pre_align()` 方法中，將原本的 Step 4-8 替換為以下實作：

```python
        # Step 4: Resize VGG level-3 (index 3 = stride-16 = 512ch) ref features to out_size
        f_vgg_ref_l3 = feats_ref[3]  # (B, 512, H_vgg, W_vgg)
        if f_vgg_ref_l3.shape[-2:] != (out_H, out_W):
            f_vgg_ref_l3 = F.interpolate(f_vgg_ref_l3, size=(out_H, out_W),
                                          mode='bilinear', align_corners=False)

        # Step 4b: Resize VGG level-2 (index 2 = stride-8 = 256ch) ref features to out_size
        f_vgg_ref_l2 = feats_ref[2]  # (B, 256, H_vgg, W_vgg) — stride8, 2× spatial res
        if f_vgg_ref_l2.shape[-2:] != (out_H, out_W):
            f_vgg_ref_l2 = F.interpolate(f_vgg_ref_l2, size=(out_H, out_W),
                                          mode='bilinear', align_corners=False)

        # Step 5: Warp both VGG scales to adverse viewpoint using the same flow field
        f_ref_warped_l3, validity_mask = warp(f_vgg_ref_l3, flow1, return_mask=True)
        f_ref_warped_l2, _ = warp(f_vgg_ref_l2, flow1, return_mask=True)

        # Step 6: Confidence = UAWarpC uncertainty → probability × boundary validity
        confidence = estimate_probability_of_confidence_interval(uncertainty1)  # (B, 1, out_H, out_W)
        confidence = confidence * validity_mask.unsqueeze(1).float()

        # Step 7: Hard mask — matches CMA paper Sec 3.3 (conf < 0.2 → zero out)
        hard_mask = (confidence >= conf_threshold).float()  # (B, 1, out_H, out_W)
        f_ref_warped_l3_masked = f_ref_warped_l3 * hard_mask  # (B, 512, out_H, out_W)
        f_ref_warped_l2_masked = f_ref_warped_l2 * hard_mask  # (B, 256, out_H, out_W)

        # Step 8: Update diagnostic attributes
        self._last_conf_mean      = float(confidence.mean().item())
        self._last_valid_ratio    = float(validity_mask.float().mean().item())
        self._last_flow           = flow1.cpu()           # (B, 2, out_H, out_W)
        self._last_confidence_map = confidence.cpu()      # (B, 1, out_H, out_W)

        return {
            'l2': f_ref_warped_l2_masked,   # (B, 256, out_H, out_W)
            'l3': f_ref_warped_l3_masked,   # (B, 512, out_H, out_W)
        }
```

**定位 old_string（現有 Step 4-8）：**
```python
        # Step 4: Resize VGG level-3 (index 3 = stride-16 = 512ch) ref features to out_size
        f_vgg_ref_l3 = feats_ref[3]  # (B, 512, H_vgg, W_vgg)
        if f_vgg_ref_l3.shape[-2:] != (out_H, out_W):
            f_vgg_ref_l3 = F.interpolate(f_vgg_ref_l3, size=(out_H, out_W),
                                          mode='bilinear', align_corners=False)

        # Step 5: Warp VGG ref features to adverse viewpoint; get validity_mask from warp()
        f_ref_warped_vgg, validity_mask = warp(f_vgg_ref_l3, flow1, return_mask=True)

        # Step 6: Confidence = UAWarpC uncertainty → probability × boundary validity
        confidence = estimate_probability_of_confidence_interval(uncertainty1)  # (B, 1, out_H, out_W)
        confidence = confidence * validity_mask.unsqueeze(1).float()

        # Step 7: Hard mask — matches CMA paper Sec 3.3 (conf < 0.2 → zero out)
        hard_mask = (confidence >= conf_threshold).float()  # (B, 1, out_H, out_W)
        f_ref_warped_vgg_masked = f_ref_warped_vgg * hard_mask

        # Step 8: Update diagnostic attributes (inside no_grad scope, cpu() sufficient)
        self._last_conf_mean      = float(confidence.mean().item())
        self._last_valid_ratio    = float(validity_mask.float().mean().item())
        self._last_flow           = flow1.cpu()           # (B, 2, out_H, out_W)
        self._last_confidence_map = confidence.cpu()      # (B, 1, out_H, out_W)

        return f_ref_warped_vgg_masked
```

---

- [ ] **Step 3: 驗證 `pre_align()` 回傳格式**

```bash
conda run -n sam_env python -c "
import torch
from segment_anything.modeling.fusion import CMAAlignment

model = CMAAlignment(pretrained_path=None)
model.eval()
img_curr = torch.randint(0, 255, (1, 3, 1024, 1024)).float()
img_ref  = torch.randint(0, 255, (1, 3, 1024, 1024)).float()
with torch.no_grad():
    out = model.pre_align(img_curr, img_ref, out_size=(64, 64))
print(type(out), {k: v.shape for k, v in out.items()})
# Expected: <class 'dict'> {'l2': torch.Size([1, 256, 64, 64]), 'l3': torch.Size([1, 512, 64, 64])}
"
```

Expected output:
```
<class 'dict'> {'l2': torch.Size([1, 256, 64, 64]), 'l3': torch.Size([1, 512, 64, 64])}
```

---

- [ ] **Step 4: Commit Task 1**

```bash
cd /home/rvl1421/SAM_research-1
git add segment-anything/segment_anything/modeling/fusion.py
git commit -m "feat: extend pre_align() to return multi-scale VGG features (l2+l3)"
```

---

## Task 2: 新增 `MultiScaleCrossAttnInjector`（取代 `MultiStageWarpedVGGInjector`）

**Files:**
- Rewrite: `segment-anything/segment_anything/modeling/vgg_adapter.py`

### 設計說明（v3：瓶頸式 Cross-Attention，適配 ACDC 1,200 張訓練集規模）

```
輸入                    KV 路徑                              Q 路徑（瓶頸）
------                  -------                              ------
multi_scale_feats       l2 (B,256,64,64)                    ViT Block output
  = {'l2', 'l3'}        l3 (B,512,64,64)                    (B,64,64,1280)
                             ↓                                    ↓
                        concat (B,768,64,64)               reshape → (B,4096,1280)
                             ↓                                    ↓
                        avg_pool (32,32)               q_down_proj (1280→256)
                        (B,768,1024)                        Q' (B,4096,256) ← 瓶頸壓縮
                         ↙          ↘                           ↓
              k_proj(768→64)   v_proj(768→64)         ┌─────────┤
              K (B,1024,64)    V (B,1024,64)           │   MultiheadAttention(
                                                       │     embed_dim=256,  ← 不是 1280
                                                       │     kdim=64, vdim=64,
                                                       │     num_heads=4
                                                       │   )
                                                       └─→ attn_out (B,4096,256)
                                                                ↓
                                                       q_up_proj (256→1280, xavier)
                                                                ↓
                                                       gate(-5.0) * q_up(attn_out)
                                                                ↓
                                                       q + gate * q_up → injected
                                                                ↓
                                                       reshape (B,64,64,1280)
```

**參數量對比（4 stages 總計）：**

| 元件 | v1 (additive) | v2 (full MHA) | v3 (bottleneck) |
|------|--------------|---------------|-----------------|
| Q projection | — | 1280×1280×4 = 6.6M | 1280×256×4 = 1.3M |
| K/V projection | 512×320×4 = 655K | 768×256×2×4 = 1.6M | 768×64×2×4 = 393K |
| MHA 內部 K/V proj + W_o | — | (256×1280×2 + 1280×1280)×4 = 9.2M | (64×256×2 + 256×256)×4 = 394K |
| q_up_proj | — | — | 256×1280×4 = 1.3M |
| adapters (v1 only) | 320×1280×4 = 1.6M | — | — |
| gates | 4 | 4 | 4 |
| **總計** | **~2.3M** | **~17.3M** | **~3.7M** |

**記憶體估計（B=1, num_heads=4, pool=32×32=1024 KV tokens）：**
- Attention matrix: 1 × 4 × 4096 × 1024 = 16M elements × 2B = 32MB (fp16) — 安全
- Q 在瓶頸維度(256)計算 attention，省去 v2 的 Q proj (1280 dim)

**超參數：**
- `vit_dim=1280`, `d_attn=256`（Q 瓶頸維度）
- `l2_channels=256`, `l3_channels=512` → concat = 768
- `d_kv=64`（K/V 投影維度）
- `pool_size=32`（pool KV 到 32×32=1024 tokens）
- `num_heads=4`（head_dim = 256/4 = 64 ≡ d_kv，對齊）
- `gate_init=-5.0`（sigmoid(-5)≈0.007；q_up_proj 為 xavier，初期注入量需更保守）

---

- [ ] **Step 1: 完整覆寫 `vgg_adapter.py`（v3：瓶頸式 Cross-Attention，~3.7M 參數）**

```python
# vgg_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCrossAttnInjector(nn.Module):
    """
    Multi-scale Bottleneck Cross-Attention Adapter。

    設計要點（v3）：
      - Q 先壓縮到 d_attn=256（瓶頸），attention 在小維度計算，避免 embed_dim=1280 的 Q
        projection（1.6M/stage）造成 ACDC 1,200 張訓練集過擬合。
      - K/V 分開投影（k_proj/v_proj），各自學習不同的特徵編碼。
      - q_up_proj：xavier 初始（非零），gate 使用 -5.0（sigmoid≈0.007）補償初期擾動。
      - 無 zero-init 的投影層阻斷梯度路徑，gate 從第一步即可學習。

    參數量：~920K/stage，4 stages 合計 ~3.7M（v2 的 1/4.7）。

    注入點：ViT-H Block [7, 15, 23, 31]（global attention blocks）
    輸入特徵：multi_scale_feats dict = {'l2': (B,256,H,W), 'l3': (B,512,H,W)}

    Diagnostics（trainer 兼容）：
        _last_inject_cos_sim  : float — 4 stage 注入前後 cosine similarity 均值
        _last_gate_val        : float — 4 stage sigmoid(gate) 均值
        _last_delta_norm_ratio: float — inject_delta_norm / vit_token_norm（訓練初期監控）
    """

    INJECT_BLOCKS: list = [7, 15, 23, 31]  # ViT-H global attention blocks

    def __init__(
        self,
        vit_dim: int = 1280,
        d_attn: int = 256,       # Q 瓶頸維度（決定 attention 計算規模）
        l2_channels: int = 256,
        l3_channels: int = 512,
        d_kv: int = 64,          # K/V 投影維度；num_heads=4, head_dim=d_attn/4=64 ≡ d_kv
        pool_size: int = 32,
        num_heads: int = 4,
        gate_init: float = -5.0, # sigmoid(-5) ≈ 0.007；q_up_proj 非零，需更保守初始
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.d_attn = d_attn
        self.pool_size = pool_size
        kv_in_channels = l2_channels + l3_channels  # 768

        num_stages = len(self.INJECT_BLOCKS)
        self._num_stages = num_stages

        # Q 瓶頸壓縮（1280 → d_attn=256）
        self.q_down_projs = nn.ModuleList([
            nn.Linear(vit_dim, d_attn)
            for _ in range(num_stages)
        ])

        # K/V 分開投影（768 → d_kv=64）
        self.k_projs = nn.ModuleList([
            nn.Linear(kv_in_channels, d_kv)
            for _ in range(num_stages)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(kv_in_channels, d_kv)
            for _ in range(num_stages)
        ])

        # 瓶頸 Cross-Attention（embed_dim=d_attn，在小維度計算）
        # MHA 內建 W_o (xavier_uniform_)，attn_out 非零，gate 梯度正常
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_attn,
                num_heads=num_heads,
                kdim=d_kv,
                vdim=d_kv,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(num_stages)
        ])

        # Q 瓶頸擴張（d_attn=256 → vit_dim=1280）— xavier 初始（非零）
        # 不使用零初始化，避免 gate 梯度斷路；初期擾動由 gate_init=-5.0 控制
        self.q_up_projs = nn.ModuleList([
            nn.Linear(d_attn, vit_dim, bias=False)
            for _ in range(num_stages)
        ])
        for proj in self.q_up_projs:
            nn.init.xavier_uniform_(proj.weight)

        # Gate（初始值 sigmoid(-5.0) ≈ 0.007；比 v2 更保守，適配非零 q_up_proj）
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_init)) for _ in range(num_stages)
        ])

        self._multi_scale_feats: dict = None
        self._stages_fired: int = 0

        _init_gate = float(torch.sigmoid(torch.tensor(gate_init)))
        self._last_inject_cos_sim: float = 1.0
        self._last_gate_val: float = _init_gate
        self._last_delta_norm_ratio: float = 0.0  # inject_delta_norm / vit_token_norm
        self._stage_cos_sims: list = [1.0] * num_stages
        self._stage_gate_vals: list = [_init_gate] * num_stages
        self._global_step: int = 0  # 供早期訓練監控使用

    def set_features(self, multi_scale_feats: dict):
        """在每次 WeatherSAM.forward 呼叫 image_encoder 前設定多尺度對齊特徵。

        Args:
            multi_scale_feats: dict with keys 'l2' (B,256,H,W) and 'l3' (B,512,H,W)
        """
        self._multi_scale_feats = multi_scale_feats
        self._stages_fired = 0

    def _make_hook(self, stage_idx: int):
        """為指定 stage 建立 forward hook closure，正確捕捉 stage_idx。"""
        def hook(module, input, output):
            return self._inject_at_stage(output, stage_idx)
        return hook

    def _inject_at_stage(self, output: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        在指定 stage 執行瓶頸式 Cross-Attention 注入。

        output shape（ViT Block 輸出）：(B, H, W, C) = (B, 64, 64, 1280)
        """
        if self._multi_scale_feats is None:
            return output

        f_l2 = self._multi_scale_feats['l2'].to(output.device, dtype=output.dtype)
        f_l3 = self._multi_scale_feats['l3'].to(output.device, dtype=output.dtype)

        if f_l3.abs().sum().item() == 0.0 and f_l2.abs().sum().item() == 0.0:
            self._stages_fired += 1
            if self._stages_fired >= self._num_stages:
                self._multi_scale_feats = None
                self._stages_fired = 0
            return output

        B, H, W, C = output.shape  # H=W=64, C=1280

        # ── Q：ViT tokens reshape + 瓶頸壓縮 ──
        q = output.reshape(B, H * W, C)                        # (B, 4096, 1280)
        q_down = self.q_down_projs[stage_idx](q)               # (B, 4096, 256)

        # ── KV：多尺度 VGG → pool → K/V 分開投影 ──
        if f_l2.shape[-2:] != (H, W):
            f_l2 = F.interpolate(f_l2, size=(H, W), mode='bilinear', align_corners=False)
        if f_l3.shape[-2:] != (H, W):
            f_l3 = F.interpolate(f_l3, size=(H, W), mode='bilinear', align_corners=False)

        f_concat = torch.cat([f_l2, f_l3], dim=1)              # (B, 768, H, W)
        f_pooled = F.adaptive_avg_pool2d(f_concat, (self.pool_size, self.pool_size))
        N_kv = self.pool_size * self.pool_size
        f_flat = f_pooled.permute(0, 2, 3, 1).reshape(B, N_kv, -1)  # (B, 1024, 768)

        k = self.k_projs[stage_idx](f_flat)   # (B, 1024, 64)
        v = self.v_projs[stage_idx](f_flat)   # (B, 1024, 64)

        # ── 瓶頸 Cross-Attention（在 d_attn=256 維度計算）──
        attn_out, _ = self.cross_attns[stage_idx](
            query=q_down,   # (B, 4096, 256)
            key=k,          # (B, 1024, 64)
            value=v,        # (B, 1024, 64)
            need_weights=False,
        )  # attn_out: (B, 4096, 256)

        # ── Q 擴張 + gate + residual ──
        delta = self.q_up_projs[stage_idx](attn_out)   # (B, 4096, 1280)
        gate = torch.sigmoid(self.gates[stage_idx])
        injected_flat = q + gate * delta               # (B, 4096, 1280)
        injected = injected_flat.reshape(B, H, W, C)

        # 診斷指標（含早期訓練監控）
        with torch.no_grad():
            cos = F.cosine_similarity(q, injected_flat, dim=-1).mean().item()
            self._stage_cos_sims[stage_idx] = cos
            self._stage_gate_vals[stage_idx] = float(gate.item())

            if stage_idx == 0:  # 只在 stage 0 計算 delta_norm_ratio，避免多次重複
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

---

- [ ] **Step 2: 驗證新 Injector 前向傳播**

```bash
conda run -n sam_env python -c "
import torch
import sys; sys.path.insert(0, 'segment-anything')
from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector

injector = MultiScaleCrossAttnInjector()
injector.eval()

# 模擬 pre_align() 輸出
feats = {
    'l2': torch.randn(1, 256, 64, 64),
    'l3': torch.randn(1, 512, 64, 64),
}
injector.set_features(feats)

# 模擬 ViT Block 輸出 (B, H, W, C)
vit_out = torch.randn(1, 64, 64, 1280)

result = injector._inject_at_stage(vit_out, stage_idx=0)
print('Output shape:', result.shape)
print('Gate val:', injector._stage_gate_vals[0])
# Expected: Output shape: torch.Size([1, 64, 64, 1280])
"
```

Expected:
```
Output shape: torch.Size([1, 64, 64, 1280])
Gate val: 0.04742...
```

---

- [ ] **Step 3: 驗證 4 stage 完整流程與自動清除**

```bash
conda run -n sam_env python -c "
import torch
import sys; sys.path.insert(0, 'segment-anything')
from segment_anything.modeling.vgg_adapter import MultiScaleCrossAttnInjector

injector = MultiScaleCrossAttnInjector()
injector.eval()

feats = {'l2': torch.randn(1, 256, 64, 64), 'l3': torch.randn(1, 512, 64, 64)}
injector.set_features(feats)

vit_out = torch.randn(1, 64, 64, 1280)
for i in range(4):
    vit_out = injector._inject_at_stage(vit_out, stage_idx=i)
    print(f'Stage {i}: fired={injector._stages_fired}, gate={injector._stage_gate_vals[i]:.4f}')

print('After all stages - _multi_scale_feats cleared:', injector._multi_scale_feats is None)
print('_last_inject_cos_sim:', injector._last_inject_cos_sim)
print('_last_gate_val:', injector._last_gate_val)
# Expected: _multi_scale_feats cleared: True
"
```

---

- [ ] **Step 4: Commit Task 2**

```bash
git add segment-anything/segment_anything/modeling/vgg_adapter.py
git commit -m "feat: replace MultiStageWarpedVGGInjector with MultiScaleCrossAttnInjector"
```

---

## Task 3: 更新 `weather_sam.py` 使用新 API

**Files:**
- Modify: `segment-anything/segment_anything/modeling/weather_sam.py:13,60,79-81,140-152`

### 設計說明

`weather_sam.py` 中有三處需要更新：
1. **import**（第 13 行）：`MultiStageWarpedVGGInjector` → `MultiScaleCrossAttnInjector`
2. **`__init__`**（第 79-81 行）：建構 `vgg_injector` 改用新 class，參數也更新
3. **`forward()`**（第 150-151 行）：`set_features()` 傳入 dict 而非 Tensor（`pre_align()` 回傳已是 dict，不需額外修改）

---

- [ ] **Step 1: 更新 import 與 `__init__`**

在 `weather_sam.py` 第 13 行，替換 import：

```python
# Before:
from .vgg_adapter import MultiStageWarpedVGGInjector

# After:
from .vgg_adapter import MultiScaleCrossAttnInjector
```

在第 79-81 行，替換 `vgg_injector` 初始化：

```python
# Before:
        self.vgg_injector = MultiStageWarpedVGGInjector(
            vgg_channels=512, vit_dim=1280, hidden_dim=320
        )

# After:
        self.vgg_injector = MultiScaleCrossAttnInjector(
            vit_dim=1280,
            d_attn=256,
            l2_channels=256,
            l3_channels=512,
            d_kv=64,
            pool_size=32,
            num_heads=4,
            gate_init=-5.0,
        )
```

---

- [ ] **Step 2: 確認 `set_features()` 呼叫（第 151 行）不需修改**

現有程式碼（第 150-151 行）：
```python
if self.use_vgg_adapter and _vgg_ref_aligned is not None:
    self.vgg_injector.set_features(_vgg_ref_aligned)
```

`_vgg_ref_aligned` 現在由 `pre_align()` 回傳，已是 dict `{'l2': ..., 'l3': ...}`。
新 `MultiScaleCrossAttnInjector.set_features()` 接受此 dict。✓ **不需修改此行。**

---

- [ ] **Step 3: 驗證 WeatherSAM 可正確建構**

```bash
conda run -n sam_env python -c "
import sys; sys.path.insert(0, 'segment-anything')
from segment_anything import build_weather_sam_vit_h

model = build_weather_sam_vit_h(checkpoint=None)
model.enable_vgg_adapter()
print('Model built successfully.')
print('vgg_injector type:', type(model.vgg_injector).__name__)
# Expected: vgg_injector type: MultiScaleCrossAttnInjector
"
```

---

- [ ] **Step 4: 端對端 forward pass 煙霧測試（需 GPU 或 CPU，B=1 即可）**

```bash
conda run -n sam_env python -c "
import torch, sys; sys.path.insert(0, 'segment-anything')
from segment_anything import build_weather_sam_vit_h

model = build_weather_sam_vit_h(checkpoint=None)
model.enable_vgg_adapter()
model.eval()

batch_input = [{
    'image': torch.randint(0, 255, (3, 1024, 1024)).float(),
    'clear_image': torch.randint(0, 255, (3, 1024, 1024)).float(),
    'text_prompts': ['road', 'sky', 'building'],
    'original_size': (1024, 1024),
    'condition_id': torch.tensor(0),
}]

with torch.no_grad():
    outputs = model(batch_input)

print('class_ids:', outputs[0]['class_ids'])
print('masks shape:', outputs[0]['masks'].shape)
# Expected: class_ids: [0, 10, 2], masks shape: torch.Size([1, 3, 1024, 1024]) (or similar)
"
```

---

- [ ] **Step 5: Commit Task 3**

```bash
git add segment-anything/segment_anything/modeling/weather_sam.py
git commit -m "feat: update WeatherSAM to use MultiScaleCrossAttnInjector with multi-scale dict API"
```

---

## Task 4: 驗證訓練迴路與 Checkpoint 兼容性

**Files:**
- Read: `segment-anything/weather_trainer.py`（只需確認診斷指標讀取方式不變）
- Modify: `segment-anything/segment_anything/build_weather_sam.py`（若有直接 import）

### 設計說明

`weather_trainer.py` 讀取以下診斷值：
- `model.vgg_injector._last_inject_cos_sim` — 新 class 有 ✓
- `model.vgg_injector._last_gate_val` — 新 class 有 ✓

`build_weather_sam.py` 不直接 import `MultiStageWarpedVGGInjector`（由 `weather_sam.py` 負責），不需修改。

但若有舊的 checkpoint 被 `torch.load()` 讀取，`MultiStageWarpedVGGInjector` 的 state_dict keys（`adapters`, `gates`）與新的 `MultiScaleCrossAttnInjector`（`kv_projs`, `cross_attns`, `out_projs`, `gates`）不同，**無法直接 load_state_dict**。此為預期行為（新架構從頭訓練）。

---

- [ ] **Step 1: 確認 trainer 診斷指標讀取相容**

```bash
conda run -n sam_env grep -n "vgg_injector\._last" segment-anything/weather_trainer.py
```

確認找到的讀取行（通常在 train/val epoch log 區段），確認讀取的是 `_last_inject_cos_sim` 和 `_last_gate_val`，兩者新 class 都有。若無則不需修改。

---

- [ ] **Step 1b: 在 trainer 加入早期訓練 delta_norm 監控（前 100 steps）**

在 `weather_trainer.py` 的 `train_epoch` 中，找到記錄 VGG injector 診斷指標的區段（grep 結果所在位置），在同一區段緊接著加入：

```python
# 早期訓練穩定性監控：inject_delta_norm / vit_token_norm
# 若比值 > 0.1，代表注入量偏大，可調低 gate_init 或增大 weight decay
if (self.use_vgg_adapter
        and hasattr(self.model.vgg_injector, '_last_delta_norm_ratio')
        and self.model.vgg_injector._global_step < 100):
    ratio = self.model.vgg_injector._last_delta_norm_ratio
    print(f"[VGG Adapter] step {self.model.vgg_injector._global_step:03d} "
          f"delta_norm_ratio={ratio:.4f} "
          f"(target < 0.1; if > 0.1 reduce gate_init from -5.0 to -7.0)")
```

此段落在前 100 個 global step 後自動靜默（`_global_step >= 100`），不影響正式訓練輸出。

---

- [ ] **Step 2: 確認 build_weather_sam.py 不需修改**

```bash
conda run -n sam_env grep -n "MultiStage\|MultiScale\|vgg_adapter" segment-anything/segment_anything/build_weather_sam.py
```

確認沒有直接 import 或 instantiate `MultiStageWarpedVGGInjector`（建構由 `weather_sam.py.__init__` 負責）。若有殘留 import，刪除之。

---

- [ ] **Step 3: 確認 `__init__.py` 中的 export（若有）**

```bash
conda run -n sam_env grep -n "MultiStage\|MultiScale\|vgg_adapter" segment-anything/segment_anything/modeling/__init__.py
```

若 `__init__.py` export 了 `MultiStageWarpedVGGInjector`，需更新為 `MultiScaleCrossAttnInjector`。

---

- [ ] **Step 4: 短暫訓練 3 steps 確認 loss 正常下降且無 CUDA OOM**

```bash
conda run -n sam_env python segment-anything/weather_trainer.py \
  --max_steps 3 \
  --output_dir /tmp/test_multiscale_crossattn \
  2>&1 | tail -30
```

預期：loss 輸出正常數值（非 nan/inf），無 CUDA OOM，inject_gate / inject_cos_sim 有記錄。

---

- [ ] **Step 5: Commit Task 4**

```bash
git add segment-anything/segment_anything/modeling/__init__.py
git add segment-anything/segment_anything/build_weather_sam.py
git commit -m "chore: clean up MultiStageWarpedVGGInjector references after cross-attn migration"
```

---

## Spec 自審結果

| 檢查項目 | 狀態 |
|----------|------|
| `pre_align()` 回傳格式從 Tensor → dict，呼叫方需更新 | ✅ Task 3 Step 2 覆蓋 |
| `set_features()` API signature 改變（Tensor → dict） | ✅ Task 2 + Task 3 覆蓋 |
| Trainer 診斷指標（`_last_inject_cos_sim`, `_last_gate_val`）相容 | ✅ Task 4 Step 1 確認 |
| Checkpoint 不相容（舊架構 key 名稱不同） | ⚠️ 預期行為，新架構從頭訓練，文件已說明 |
| `need_weights=False` 傳入 MHA 避免 attention weight 計算開銷 | ✅ Task 2 Step 1 已實作 |
| 全零 sentinel 特徵跳過注入邏輯 | ✅ Task 2 Step 1 已實作 |
| 空間尺寸 mismatch 時 `F.interpolate` 對齊 | ✅ Task 2 Step 1 l2/l3 均有 guard |
| **out_proj 零初始化切斷 gate 梯度（v1 bug）** | ✅ v3 改為 `q + gate * q_up(attn_out)`，q_up xavier init，gate 梯度即時 |
| **avg_pool 16 丟失動態物件細節（v1 問題）** | ✅ v3 pool_size 改為 32×32=1024 tokens |
| **K/V 同源限制表達（v1 問題）** | ✅ v3 改為 `k_proj` / `v_proj` 分開投影 |
| **v2 ~17M 參數過多（ACDC 1,200 張過擬合風險）** | ✅ v3 瓶頸設計降至 ~3.7M（Q proj: 1280→256→1280 vs v2 的 1280→1280）|
| **MHA 初期擾動（q_up xavier 非零）** | ✅ gate_init: -3.0→-5.0（sigmoid≈0.007），Task 4 加入前 100 steps 監控 |

---

## 預期效果

| 指標 | v1（additive MLP） | v3（bottleneck cross-attn） |
|------|------|---------|
| 注入機制 | 加法 MLP (512→320→1280) | 瓶頸 Cross-Attention（Q: 1280→256→1280，K/V 分開） |
| 可訓練參數 | ~2.3M | ~3.7M（適合 ACDC 1,200 張）|
| 參考尺度 | 單尺度（VGG pool4, stride16） | 雙尺度（pool3 stride8 + pool4 stride16） |
| 動態物件適應能力 | 弱（無注意力選擇） | 較佳（attention 選擇性忽略未對齊區域） |
| Gate 初始梯度 | 正常 | ✅ 正常（xavier q_up_proj，gate_init=-5.0，delta_norm監控） |
| 推論記憶體增量 | ~0 (純 MLP) | ~32MB/sample（4heads × 4096×1024 KV tokens）|
| K/V 表達自由度 | — | k_proj / v_proj 分開，表達更豐富 |
