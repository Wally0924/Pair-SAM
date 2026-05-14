# NCAR2026 — E27 權威數據總表

**Date:** 2026-05-14
**Source:** `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json`
**Checkpoint:** `best_E27_mIoU65.68_LR4.0e-05.pth`（train_log running-avg 65.68%，confusion-matrix re-eval 65.51%）
**Eval split:** ACDC val（406 張：fog=100, rain=100, snow=100, night=106）

---

## 核心數字（直接抄進論文用）

| 項目 | 值 | 寫進論文時的措辭 |
|------|-----|-----------------|
| Overall val mIoU | **65.51 %** | 「65.5 %」或「65.51 %」 |
| Trainable params | 24,534,329 | 「24.5 M」 |
| Total params | 823,618,244 | 「823.6 M」 |
| Trainable ratio | 2.978 % | 「2.98 %」 |
| Frozen params | 799,083,915 | 「799.1 M」 |
| Train epochs (actual) | 37 | 「trained for 37 epochs」 |
| Best-checkpoint epoch | 27 | 「model selection at epoch 27」 |
| Initial LR | 5×10⁻⁵ | as-is |
| Image resolution | 1024×1024 | as-is |
| GPU | 24 GB | as-is |

---

## Per-Condition mIoU（ACDC val）

| Condition | mIoU (%) |
|-----------|---------:|
| Fog       | 70.56 |
| Rain      | 64.35 |
| Snow      | 69.30 |
| Night     | 48.85 |
| **All**   | **65.51** |

---

## Per-Class IoU（ACDC val Overall column；19 類）

| Class         | IoU (%) |
|---------------|--------:|
| road          | 95.43 |
| sidewalk      | 80.80 |
| building      | 88.40 |
| wall          | 61.79 |
| fence         | 52.56 |
| pole          | 62.92 |
| traffic light | 69.89 |
| traffic sign  | 65.10 |
| vegetation    | 89.63 |
| terrain       | 54.42 |
| sky           | 97.80 |
| person        | 64.83 |
| rider         |  4.88 |
| car           | 87.61 |
| truck         | 62.07 |
| bus           | 51.80 |
| train         | 68.46 |
| motorcycle    | 41.56 |
| bicycle       | 44.72 |

**全 4 條件 × 19 類完整矩陣** 見 `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md`。

---

## 對照論文 baselines（同 ACDC val，從各論文表格抄出）

| Method | Backbone | Trainable | Regime | Overall val mIoU | 引用 |
|--------|----------|----------:|--------|-----------------:|------|
| SegFormer-B5 (Source only) | SegFormer-B5 | 85 M | Cityscapes only | **56.6** | CMA Tab. 6 |
| URMA | SegFormer-B5 | 85 M | model adapt | **63.2** | CMA Tab. 6 |
| Refign-DAFormer | DAFormer | 85 M | UDA + ref + WarpC | **65.0** | Refign Tab. 4 row 6 |
| **WeatherSAM (Ours, E27)** | **SAM ViT-H frozen** | **24.5 M** | **supervised + ref** | **65.51** | **本文** |
| CMA | SegFormer-B5 | 85 M | model adapt + ref + contrastive | **67.2** | CMA Tab. 5 row 7 / Tab. 6 |

**讀法：**
- 我們的 65.51 比 Refign 高 **+0.51 mIoU**（「marginally above」，而非 "beats"）
- 我們的 65.51 比 CMA 低 **−1.69 mIoU**（誠實 gap）
- 可訓練參數只佔 CMA 的 24.5/85 ≈ **29 %**（或 SAM ViT-H 的 2.98 %）

---

## Per-Module trainable breakdown（給 Table 1 / 參數表用）

| Module | Total | Trainable | Frozen |
|--------|------:|----------:|:------:|
| SAM ViT-H image encoder | 637.0 M | 0 | ✓ |
| CLIP text encoder (frozen) + projection (trainable) | 151.4 M | 0.13 M | partial |
| CMAAlignment (VGG-16 + UAWarpC) | 10.8 M | 0 | ✓ |
| Cross-Attention Adapter (4 injection points) | 17.3 M | 17.3 M | — |
| TwoWayTransformer (fine-tuned at 1/20× LR) | 3.3 M | 3.3 M | — |
| Class tokens + hypernetworks + upscaling | 2.7 M | 2.7 M | — |
| Dense positional encoding `pe_layer` | 1.0 M | 1.0 M | — |
| Logit Refinement + Condition Encoder + Prompt Encoder | < 5 K | < 5 K | — |
| **Total** | **823.6 M** | **24.5 M (2.98 %)** | 799.1 M |

---

## E18 vs E27 對照（為了透明，保留歷史軌跡）

| Metric | E18 (舊) | E27 (新，論文採用) | Δ |
|--------|---------:|------------------:|---:|
| Overall mIoU | 64.91 | **65.51** | +0.60 |
| Fog | 67.33 | **70.56** | +3.23 |
| Rain | 62.87 | **64.35** | +1.48 |
| Snow | 68.51 | **69.30** | +0.79 |
| Night | 48.35 | **48.85** | +0.50 |

E27 在 fog 與 rain 改善最多；night 雖然仍是最弱條件，也有微幅上升。

---

## 訓練設定（§4.1 用）

```
Optimizer:        AdamW
Initial LR:       5e-5 (cosine decay after 5-epoch linear warm-up)
Weight decay:     1e-2
Effective batch:  4 (batch 1 × grad_accum 4)
AMP:              ✓
Grad clip:        1.0
Epochs trained:   37   (model selection at epoch 27 via val mIoU)
Image size:       1024 × 1024
Hardware:         single 24 GB GPU
Gate warm-up:     N_g = 3 epochs (gates frozen)
```

---

## 不可寫進論文的數字 / 措辭（已從 audit plan 移除）

- ❌ 「80 epochs」（實際 37）
- ❌ 「64.91 %」（舊 E18 數字）
- ❌ 「consistent mIoU gains across all four conditions」（night 48.85 不 consistent）
- ❌ 「improvement is most pronounced on fog and night」（沒有 baseline 的 per-condition 數字可以做 delta）
- ❌ 「beats / outperforms CMA」（我們是 65.51，CMA 是 67.2，我們輸 1.7 mIoU）
- ❌ 「essentially on par with Refign」（E27 +0.5 mIoU 已可改寫為「marginally above」）
- ❌ "+X.X over CMA 69.1 %"（69.1 是 CMA 的 **test** mIoU，不是 val）

---

## 引用論文表格的精確位置（reviewer 會去查）

- Refign 65.0 % = WACV'23 Bruggemann et al. **Table 4 row 6**（full model: ALIGN ✓ P_R ✓ M ✓ s ✓ R-ad ✓）
- CMA 67.2 % = ICCV'23 Bruggemann et al. **Table 5 row 7**（full model）or **Table 6 row "CMA contrastive"**
- Source SegFormer 56.6 % = CMA ICCV'23 **Table 6 row 1**
- URMA 63.2 % = CMA ICCV'23 **Table 6 row 2**
- DAFormer baseline per-condition (fog 67.9 / night 34.8) = Refign WACV'23 **Fig. 5 caption** — **注意：這是 DAFormer baseline，不是 Refign-DAFormer**

---

## 後續行動清單

1. ✅ E27 重新評估（已完成，commit `020d237`）
2. ✅ Audit plan 更新（commit 待補）
3. ⬜ 把這份數據塞回 main.tex（依 audit plan Part F 執行順序）
4. ⬜ §4.1 修正「80 epochs」措辭
5. ⬜ E27 重新跑 E4 / E5 後續若視覺差異不大，原檔即可用（已 commit）
6. ⬜ （可選）跑 ACDC test set 取得官方 test mIoU
