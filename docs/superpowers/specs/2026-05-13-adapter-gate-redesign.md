# Adapter Gate Redesign + Train Log 補齊

**日期：** 2026-05-13
**分支：** feat/image-pair-fusion
**問題根因：** inject_gate 全程 ≈ 0.007，Adapter 實際貢獻 < 1%；
              Cross-Attention Q 來自 ViT token，導致 delta 與 q 高度相關（inject_cos_sim ≈ 0.79–0.83）。

---

## 背景與動機

### 已觀察到的問題

| 指標 | 觀測值 | 問題 |
|------|--------|------|
| `val_inject_gate` | 0.00675 → 0.00714（29 epoch）| Gate 幾乎靜止，Adapter 貢獻 < 1% |
| `val_inject_cos_sim` | 0.79–0.83 | delta 與 ViT token 高度相關，未注入新資訊 |
| `val_fusion_cos_sim` | 恆為 0.0 | 死欄位，指標從未被計算 |
| `inject_delta_norm_ratio` | 已計算但未寫入 CSV | 缺乏梯度規模監控 |

### 根本原因分析

**Gate 靜止：**
`sigmoid(-5.0) ≈ 0.007`，且 sigmoid 在 -5 附近梯度 ≈ 0.007。
即使 loss 對 gate 有梯度，每步推動 gate 也微乎其微；
pre-hook 注入後信號還需穿過整個 ViT 自注意力層才到達 loss，梯度路徑更長。

**inject_cos_sim 偏高：**
Cross-Attention 以 ViT token 的瓶頸投影作為 Q，
attention 傾向讓 VGG K/V 迎合 ViT 已有表示，
使 delta = f(q, VGG) ≈ f(q)，未能注入真正異質的天氣補償信號。
（等同於 cos(q, q + gate*delta) ≈ 1，數學上被迫高相似）

---

## 解決方案

### 改動一：Gate 機制 — sigmoid → softplus

**變更範圍：** `segment_anything/modeling/vgg_adapter.py`

```
Before:
    gate_init = -5.0
    self.gates[i] = nn.Parameter(torch.tensor(-5.0))
    gate = torch.sigmoid(self.gates[stage_idx])

After:
    _gate_init = log(exp(0.05) - 1)  # ≈ -2.94, softplus(-2.94) ≈ 0.05
    self.gates[i] = nn.Parameter(torch.tensor(_gate_init))
    gate = F.softplus(self.gates[stage_idx])
```

**為什麼 softplus 更好：**
- `softplus(x)` 在 x=-2.94 處梯度 = sigmoid(-2.94) ≈ 0.05（比舊設計高 7×）
- 正域近似線性，允許 gate > 1.0（放大 delta），不受 sigmoid 上界 1.0 限制
- 初始值 0.05 讓訓練第一步就有可觀測的 Adapter 貢獻，但不影響訓練穩定性

**診斷屬性更新：**
`_last_gate_val` 的語意從 sigmoid 值改為 softplus 值，數值範圍 [0, ∞)，
trainer 的讀取方式不變。

---

### 改動二：Q 解耦 — Cross-Attention → SAM-Adapter 風格 MLP

**變更範圍：** `segment_anything/modeling/vgg_adapter.py`

**核心問題：** 現有架構以 ViT token 作為 Cross-Attention 的 Q，
使 delta 在數學上必然與 q 相似，無法注入真正差異化的天氣補償信號。

**新設計（SAM-Adapter 風格）：**
```
VGG feats (l2: B,256,H,W) + (l3: B,512,H,W)
    → concat → adaptive_avg_pool(32×32) → flatten (B, 1024, 768)
    → MLP_down: (768 → d_hidden=256), GELU
    → MLP_up:   (256 → vit_dim=1280)
    → spatial_mean → (B, 1, 1280) → expand → (B, 4096, 1280) = delta
```

**移除的組件：**
- `q_down_projs`（ViT token 瓶頸壓縮，1280→256）
- `cross_attns`（MultiheadAttention）
- `q_up_projs`（瓶頸擴張，256→1280）
- `k_projs`、`v_projs`（獨立的 K/V 投影）

**新增的組件（per-stage）：**
- `vgg_mlp_downs`：`Linear(768, 256)` × 4 stages
- `vgg_mlp_ups`：`Linear(256, vit_dim=1280)` × 4 stages
  - `MLP_up` 最後一層：zero-init weight，避免初期擾動；gate warmup 額外保護

**殘差結構保持不變：**
```python
injected_flat = q + gate * delta   # q 來自 ViT token，delta 純來自 VGG
injected = injected_flat.reshape(B, H, W, C)
```

**預期效果：**
- `inject_cos_sim` 應從 0.79 下降至 0.3–0.6（delta 來自純 VGG，與 q 無依賴關係）
- 參數量從 ≈3.7M 降至 ≈(768×256+256×1280)×4 ≈ 2.1M，降低過擬合風險

---

### 改動三：Gate Warmup（`weather_trainer.py`）

前 `warmup_gate_epochs=3` 個 epoch 凍結 gate 參數，
讓 main decoder 先建立穩定的分類邊界，再開放 Adapter 學習補償信號。

**實作方式：**
```python
# Trainer.__init__：識別 gate 參數
self._gate_params = [
    p for n, p in model.named_parameters()
    if 'vgg_injector.gates' in n
]

# 每 epoch 開始時切換
is_warmup = (epoch < self.warmup_gate_epochs)
for p in self._gate_params:
    p.requires_grad_(not is_warmup)
```

`warmup_gate_epochs` 預設 3，透過 `train.py` argparse 傳入。

---

### 改動四：Train Log 補齊（`weather_trainer.py` + `train.py`）

#### 4-1 新增 AverageMeter（trainer）

| 新欄位 | 資料來源 |
|--------|---------|
| `inject_delta_norm_ratio` | `_injector._last_delta_norm_ratio`（已計算） |
| `head_delta_norm` | `losses['head_delta_norm']`（已計算） |
| `inject_gate_s0` ~ `s3` | `_injector._stage_gate_vals[0..3]` |
| `inject_cos_s0` ~ `s3` | `_injector._stage_cos_sims[0..3]` |

#### 4-2 history dict 新增欄位（`train.py`）

```python
"train_inject_delta_norm": train_metrics.get("inject_delta_norm", 0.0),
"val_inject_delta_norm":   val_metrics.get("inject_delta_norm",   0.0),
"train_head_delta_norm":   train_metrics.get("head_delta_norm",   0.0),
"val_head_delta_norm":     val_metrics.get("head_delta_norm",     0.0),
# per-stage gate
"train_inject_gate_s0": ..., "val_inject_gate_s0": ...,
"train_inject_gate_s1": ..., "val_inject_gate_s1": ...,
"train_inject_gate_s2": ..., "val_inject_gate_s2": ...,
"train_inject_gate_s3": ..., "val_inject_gate_s3": ...,
# per-stage cos_sim
"train_inject_cos_s0": ..., "val_inject_cos_s0": ...,
# ... s1, s2, s3 同上
```

#### 4-3 移除死欄位

`train_fusion_cos_sim` / `val_fusion_cos_sim` 從 history dict 移除（全程恆為 0.0）。

---

## 變更摘要

| 檔案 | 改動性質 |
|------|---------|
| `segment_anything/modeling/vgg_adapter.py` | Gate: sigmoid→softplus；Q路徑: cross-attn→MLP |
| `weather_trainer.py` | 新增 gate warmup；新增 per-stage AverageMeter 讀取 |
| `train.py` | history dict 新增 8 個診斷欄位；移除 2 個死欄位 |

---

## 成功驗證標準

訓練後，以下指標應有明顯改善：

1. `val_inject_gate` 在 epoch 10 前應超過 0.05，epoch 20 前超過 0.1
2. `val_inject_cos_sim` 應從 0.79 降至 0.5 以下（delta 與 q 解耦）
3. `val_inject_delta_norm_ratio` 應從 < 0.01 升至 0.05–0.2 範圍
4. `val_miou` 維持或超過 60.8%（確認改動不造成退步）
5. 所有 per-stage gate/cos 欄位在 CSV 中正確記錄（非零）

---

## 不在本次範圍內

- Per-channel gate（1280-dim）：等 mIoU > 65% 再考慮
- 消融實驗（去掉 Adapter 的 baseline 訓練）：獨立實驗
- KV 壓縮比調整（現有 pool_size=32 保持不動）
