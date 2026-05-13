# Cross-Attention Injector v5 Redesign

**日期：** 2026-05-13
**分支：** feat/image-pair-fusion
**前因：** v4 MLP（SAM-Adapter 風格）因 MLP_up zero-init + 小 gate 造成梯度死鎖，42 epoch 後 val_mIoU 僅 54.0%（v3 Cross-Attn 達 60.8%）。

---

## 問題診斷

### v4 失敗根因

| 觀測 | 數值（epoch 42）| 根因 |
|------|----------------|------|
| val_inject_gate | 0.050 → 0.060 | 梯度死鎖：delta≈0 → gate 梯度≈0 |
| val_inject_cos_sim | 1.0 → 0.86 | delta≈0，q+0*delta≈q，cos→1 |
| val_mIoU | 54.0% | Adapter 全程零貢獻，main decoder 獨立運作 |

**梯度死鎖機制：**
```
MLP_up weight = 0  →  delta = 0
delta = 0          →  gate 梯度 ≈ gate × downstream ≈ 0.05 × ~0 ≈ 0
gate 梯度 ≈ 0      →  MLP_up 梯度 = gate × grad ≈ 極小
→ 雙方互相阻礙，42 epoch 仍無法脫離初始狀態
```

### v3 的殘留問題（本次設計需解決）

| 觀測 | 數值 | 問題 |
|------|------|------|
| val_inject_gate | 0.007（sigmoid(-5)）| gate 幾乎不開 |
| val_inject_cos_sim | 0.79–0.83 | delta 與 q 方向相似，未注入差異化資訊 |

---

## 設計方案：MultiScaleCrossAttnInjector v5

### 架構概覽

```
VGG l2 (B,256,H,W) ──┐
                       ├─ concat(768,H,W) → pool(32×32) → flat(B, 1024, 768)
VGG l3 (B,512,H,W) ──┘
                                │
                        k_proj(768→256)   v_proj(768→256)   [Xavier init]
                                │                │
                        K(B,1024,256)     V(B,1024,256)
                                       │
ViT token (B,H*W,1280) ── .detach() ──┤   Q 不壓縮，全維 1280
                                       Q(B,H*W,1280)
                                       │
                  MHA(embed_dim=1280, kdim=256, vdim=256, num_heads=4)
                                       │
                              delta(B,H*W,1280)
                                       │
injected = q + softplus(gate) * delta
         ↑                   ↑
    有梯度（殘差）        初始 ≈ 0.05，無上界
```

**注入點：** ViT-H block [7, 15, 23, 31]，pre-hook
**每 stage 獨立：** k_proj × 4、v_proj × 4、cross_attns × 4、gates × 4

---

## 關鍵設計決策

### 1. Q = ViT tokens + stop_gradient

```python
q = output.reshape(B, H * W, C)          # 有梯度，用於殘差
Q = q.detach()                            # 無梯度，僅用於 attention 計算
delta = self.cross_attns[stage_idx](Q, K, V)
injected_flat = q + gate * delta
```

**為何 stop_gradient 能降低 cos_sim：**
- v3 問題：optimizer 可透過 Q→attn→delta 的梯度路徑，主動推動 delta 對齊 q（最小化 loss 的捷徑）
- stop_gradient 後：optimizer 無法利用此路徑，delta 只能捕捉 VGG 中 ViT **未有**的資訊
- Q 的數值仍來自 ViT token（場景感知），attention 仍能選擇語意互補的 VGG 特徵

### 2. Q 不壓縮（全維 1280）

- v3 使用 q_down_projs（1280→256）壓縮 Q，損失語意細節
- v5 移除 Q bottleneck，保留完整 1280-dim 語意用於 attention selection
- K/V 側壓縮至 256（768→256），控制整體參數量

**參數量估算：**

| 組件（per stage）| 參數數 |
|----------------|--------|
| k_proj: Linear(768, 256) | 196,608 |
| v_proj: Linear(768, 256) | 196,608 |
| MHA(1280, kdim=256, vdim=256, heads=4) | ≈ 1,638,400 + 65,536×2 |
| gate: scalar Parameter | 1 |
| **Per-stage 合計** | **≈ 2.1M** |
| **4 stages 合計** | **≈ 8.5M** |

### 3. Xavier init 取代 zero-init

```python
# v4（死鎖）
nn.init.zeros_(proj.weight)

# v5（正確）
# MHA 使用 PyTorch 預設 Xavier init（Kaiming uniform）
# k_proj、v_proj 同樣使用預設 Xavier
```

穩定性由 **gate warmup**（前 3 epoch 凍結 gate）保障，而非 zero-init。
- Gate 凍結期：delta 有梯度，MHA/k_proj/v_proj 自由學習
- Gate 解凍後：delta 已有意義，gate 有效梯度信號，能正常開啟

### 4. Gate：softplus + warmup（沿用 v4 設計）

```python
_DEFAULT_GATE_INIT = math.log(math.exp(0.05) - 1)  # ≈ -2.9444
gate = F.softplus(self.gates[stage_idx])             # 初始 ≈ 0.05
```

- softplus 梯度（在 -2.94）≈ 0.05，比 sigmoid(-5) 的 0.007 強 7×
- 無上界，允許 gate > 1.0

---

## Diagnostics（Trainer 相容，不需修改 trainer）

| 屬性 | 說明 |
|------|------|
| `_last_inject_cos_sim` | cos(q, injected) 4 stage 均值 |
| `_last_gate_val` | softplus(gate) 4 stage 均值 |
| `_last_delta_norm_ratio` | ‖gate×delta‖ / ‖q‖ |
| `_stage_gate_vals[0..3]` | per-stage gate 值 |
| `_stage_cos_sims[0..3]` | per-stage cos(q, injected) |

---

## 變更範圍

| 檔案 | 動作 | 說明 |
|------|------|------|
| `segment-anything/segment_anything/modeling/vgg_adapter.py` | 覆寫 | v5 架構 |
| `segment-anything/segment_anything/modeling/weather_sam.py` | 修改 | injector 參數名更新 |
| `segment-anything/tests/test_vgg_adapter_pre_hook.py` | 覆寫 | v5 API 測試 |
| `segment-anything/weather_trainer.py` | **不動** | diagnostic 屬性相容 |
| `segment-anything/train.py` | **不動** | CSV 欄位相容 |

---

## 測試清單

1. `test_no_mlp_modules` — vgg_mlp_downs / vgg_mlp_ups 不存在
2. `test_has_cross_attn_modules` — k_projs / v_projs / cross_attns 各 4 個
3. `test_no_q_bottleneck` — q_down_projs 不存在（Q 全維）
4. `test_gate_initial_value_approx_0_05` — softplus(gates[0]) ≈ 0.05 ± 0.005
5. `test_xavier_init_not_zero` — k_proj weight max > 0
6. `test_inject_shape_preserved` — 輸出 (B,H,W,C) 不變
7. `test_delta_driven_by_vgg_not_vit` — 固定 Q，不同 VGG K/V → delta 不同
8. `test_vit_q_detached_no_grad` — detach 後 Q.requires_grad == False
9. `test_diagnostics_updated_after_all_stages` — 4 stage 執行後 diagnostic 屬性更新
10. `test_pre_hook_returns_tuple` — hook 回傳 tuple，shape 正確
11. `test_make_hook_post_still_exists` — ablation hook 保留

---

## 成功驗證標準

訓練後（epoch 20 為節點）：

| 指標 | 目標 |
|------|------|
| val_inject_gate | epoch 10 前 > 0.08，epoch 20 前 > 0.15 |
| val_inject_cos_sim | epoch 20 前降至 < 0.70（v3 0.79，MLP 路徑改善）|
| val_inject_delta_norm_ratio | > 0.05（adapter 有實質貢獻）|
| val_mIoU | epoch 30 ≥ 61%（超越 v3）|

---

## 不在本次範圍內

- LayerNorm on delta（方案三）：待 v5 mIoU > 63% 後考慮
- Per-channel gate（1280-dim）：等 mIoU > 65%
- 消融實驗（無 Adapter baseline）：獨立實驗
