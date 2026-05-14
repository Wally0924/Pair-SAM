# NCAR2026 — Experiments 完整修訂計畫（E27 權威版）

**Date:** 2026-05-14
**Source of truth:** `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.json`
**Checkpoint:** `best_E27_mIoU65.68_LR4.0e-05.pth`（train_log running-avg 65.68 %，confusion-matrix re-eval 65.51 %）
**Eval split:** ACDC val（406 張：fog=100, rain=100, snow=100, night=106）
**Audit basis:** `research-paper-writing` skill — three core questions（better than baselines / which design choices contribute / how far does it generalize）+ claim-evidence binding from `references/paper-review.md`

**本文件取代並合併：**
- `NCAR2026_E27_authoritative_numbers.md`（權威數字表）
- `NCAR2026_experiments_rewrite_plan.md`（修訂審查清單）

---

# Part 0 — 權威數據（Single Source of Truth）

## 0.1 核心數字

| 項目 | 值 | 寫進論文時的措辭 |
|------|-----|-----------------|
| Overall val mIoU | **65.51 %** | 「65.5 %」或「65.51 %」 |
| Trainable params | 24,534,329 | 「24.5 M」 |
| Total params | 823,618,244 | 「823.6 M」 |
| Trainable ratio | 2.978 % | 「2.98 %」 |
| Frozen params | 799,083,915 | 「799.1 M」 |
| Train epochs (actual) | 37 | 「trained for 37 epochs」 |
| Best-checkpoint epoch | 27 | 「model selection at epoch 27 by val mIoU」 |
| Initial LR | 5×10⁻⁵ | as-is |
| Image resolution | 1024×1024 | as-is |
| GPU | 24 GB | as-is |

## 0.2 Per-Condition mIoU（ACDC val）

| Condition | mIoU (%) | Samples |
|-----------|---------:|--------:|
| Fog       | 70.56 | 100 |
| Rain      | 64.35 | 100 |
| Snow      | 69.30 | 100 |
| Night     | 48.85 | 106 |
| **All**   | **65.51** | **406** |

## 0.3 Per-Class IoU（ACDC val Overall column；19 類）

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

全 4 條件 × 19 類完整矩陣見 `docs/experiments/v15-eval-2026-05-14/e1_acdc_val_results.md`。

## 0.4 對照論文 baselines（同 ACDC val）

| Method | Backbone | Trainable | Regime | Overall val mIoU | 引用 |
|--------|----------|----------:|--------|-----------------:|------|
| SegFormer-B5 (Source only) | SegFormer-B5 | 85 M | Cityscapes only | **56.6** | CMA Tab. 6 |
| URMA | SegFormer-B5 | 85 M | model adapt | **63.2** | CMA Tab. 6 |
| Refign-DAFormer | DAFormer | 85 M | UDA + ref + WarpC | **65.0** | Refign Tab. 4 row 6 |
| **WeatherSAM (Ours, E27)** | **SAM ViT-H frozen** | **24.5 M** | **supervised + ref** | **65.51** | **本文** |
| CMA | SegFormer-B5 | 85 M | model adapt + ref + contrastive | **67.2** | CMA Tab. 5 row 7 / Tab. 6 |

**讀法：**
- WeatherSAM 比 Refign **高 +0.51 mIoU**（措辭：「marginally above」，**不是** "beats"）
- WeatherSAM 比 CMA **低 −1.69 mIoU**（誠實 gap）
- 可訓練參數只佔 CMA 的 24.5/85 ≈ **29 %**，或 SAM ViT-H 的 **2.98 %**

## 0.5 Per-Module Trainable Breakdown

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

## 0.6 訓練設定（§4.1 用）

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

## 0.7 E18 vs E27 對照（為了透明，保留歷史軌跡）

| Metric | E18 (舊，已棄用) | **E27 (新，論文採用)** | Δ |
|--------|---------:|------------------:|---:|
| Overall mIoU | 64.91 | **65.51** | +0.60 |
| Fog | 67.33 | **70.56** | +3.23 |
| Rain | 62.87 | **64.35** | +1.48 |
| Snow | 68.51 | **69.30** | +0.79 |
| Night | 48.35 | **48.85** | +0.50 |

E27 在 fog 與 rain 改善最多；night 雖然仍是最弱條件，也有微幅上升。

## 0.8 引用論文表格的精確位置（reviewer 會去查）

- **Refign 65.0 %** = WACV'23 Bruggemann et al. **Table 4 row 6**（full model: ALIGN ✓ P_R ✓ M ✓ s ✓ R-ad ✓）
- **CMA 67.2 %** = ICCV'23 Bruggemann et al. **Table 5 row 7**（full model）or **Table 6 row "CMA contrastive"**
- **Source SegFormer 56.6 %** = CMA ICCV'23 **Table 6 row 1**
- **URMA 63.2 %** = CMA ICCV'23 **Table 6 row 2**
- **DAFormer baseline per-condition (fog 67.9 / night 34.8)** = Refign WACV'23 **Fig. 5 caption** — ⚠️ 注意：這是 DAFormer baseline，**不是** Refign-DAFormer

## 0.9 不可寫進論文的數字 / 措辭（負面清單）

- ❌ 「80 epochs」（實際 37）
- ❌ 「64.91 %」（舊 E18 數字）
- ❌ 「consistent mIoU gains across all four conditions」（night 48.85 不 consistent）
- ❌ 「improvement is most pronounced on fog and night」（沒有 baseline 的 per-condition 數字可以做 delta）
- ❌ 「beats / outperforms CMA」（我們是 65.51，CMA 是 67.2，輸 1.7 mIoU）
- ❌ 「essentially on par with Refign」（E27 +0.5 mIoU 已可改寫為「marginally above」）
- ❌ "+X.X over CMA 69.1 %"（69.1 是 CMA 的 **test** mIoU，不是 val）

---

# Part A — 你目前 §4 草稿的稽核

## A.1 強項（保留不動）

- ✅ **Honest framing.** With E27 numbers (65.51 %) the framing shifts from "essentially on par with Refign" to **"marginally above Refign-DAFormer (65.0 %)"** and the gap to CMA tightens from −2.3 to **−1.69 mIoU**. Both stay reviewer-defensible without overclaiming.
- ✅ **Parameter-efficiency angle.** 24.5 M / 823.6 M = 2.98 % is a clean, defensible primary contribution; you do not need to outperform CMA on accuracy to publish.
- ✅ **Per-condition / per-class standalone analysis.** Refusing to fabricate per-condition baseline numbers is correct — Refign Tab 4 and CMA Tab 5 only report *overall* val mIoU.
- ✅ **Limitations paragraph (§4.7).** Three honest scope statements (val-only, regime asymmetry, GNSS pairing requirement).
- ✅ **Self-review claim-evidence map.** You already flagged `+X.X over CMA`, `most pronounced on fog and night`, `pre-hook preserves residual stream` as not-fully-supported. This is where the cross-section work in Part B lives.

## A.2 需要修正

### Issue R1 — Outline vs prose 章節順序不一致

舊草稿的 outline（上方）：
> §4.1 Setup → §4.2 **Main Results** → §4.3 Per-Condition Analysis → §4.4 **Parameter Efficiency** → §4.5–4.7

舊草稿的 prose（下方）：
> §4.1 Setup → §4.2 **Parameter Efficiency** → §4.3 **Main Results** → §4.4 Per-Class Analysis → §4.5–4.7

**建議：** 採 **prose ordering**——*Parameter Efficiency 在前*——因為它領先最強的貢獻，並把後續 Main Results 表重新定位為「2.98 % trainable 換來什麼樣的準確度？」。這是教科書級的 *narrative framing*：當你 Y 軸贏不了，就先錨定 X 軸，再說明 Y 在給定 X 下是 competitive。

→ **Action:** outline 跟著 prose 走，最終順序見 **Part C**。

### Issue R2 — Table 編號錯亂

舊 prose 引用：`Table 1`（從未出現）、`Table 2 = param`、`Table 3 = main val`、`Table 4 = per-condition + per-class`。
舊正式表格清單卻把 `Table 3` 標為 main val、`Table 2` 標為 param。沒有 Table 1，舊的 ACDC test placeholder 被靜默替換。

**建議重新編號（依首次引用順序）：**

| New label | Content | First-cited in |
|---|---|---|
| **Table 1** | Trainable parameter breakdown (24.5 M / 823.6 M) | §4.2 Parameter Efficiency |
| **Table 2** | Main val comparison (Source / Refign / CMA / WeatherSAM, overall + 我們的 per-condition 4 欄) | §4.3 Main Results |
| **Table 3** | Per-class IoU breakdown for WeatherSAM (19 classes × 4 conditions, or aggregate) | §4.4 Per-Class Analysis |

把舊「Table 4 per-condition」的 4 個數字 **合併進 Table 2 的 WeatherSAM 列** 作為 4 個額外欄位；per-class 獨立成 Table 3。這把 4 表縮為 3 表，並避開「我們有 per-condition baselines 沒有」的尷尬——baseline 那幾格直接寫 `—`。

→ **Action:** 重編 + 合併；刪除多餘的舊 Table 4。

### Issue R3 — Abstract「consistent mIoU gains across all four conditions」是 overclaim

E27 per-condition：fog 70.56 / rain 64.35 / snow 69.30 / night **48.85**。Night 比其他三條件低約 20 mIoU。「consistent」這個詞撐不住，reviewer 一看表就會抓。

**建議改成下列其一：**
- *"competitive mIoU on ACDC across all four conditions"*，或
- *"usable mIoU on ACDC across all four conditions, with the night split as the remaining bottleneck"*

第二個版本 **front-loads the limitation**，reviewer 通常會給 credit。

### Issue R4 — §1 P3「improvement is most pronounced on fog and night」結構性不成立

E27 val: fog=70.56, rain=64.35, snow=69.30, night=48.85，**且沒有任何 baseline per-condition 數字可減**。標榜「+X.X on fog/night」沒有減法可做。即使當 *standalone observation* 寫，事實也相反——night 是最差條件，與「most pronounced improvement」剛好對立。

→ **Action:** 直接刪除這句；不替換為其他「most pronounced」claim。Per-condition mIoU 在 Table 2 standalone 呈現即是誠實表達。

---

# Part B — 跨章節 Claim 修正（X-1 ~ X-6）

六個全文範圍的 claim 必須更新——它們是在 `XX.X` placeholder 時期寫的，現在跟真實數據矛盾。

### ☐ Edit X-1 — English Abstract last sentence

**Current:**
> *"This yields consistent mIoU gains on ACDC across all four conditions while only a small fraction of SAM's parameters is trained."*

**After:**
> *"This yields competitive mIoU on ACDC across all four conditions while only **2.98 %** of SAM's parameters are trained."*

**Rationale:** 「consistent」→「competitive」；把 headline number `2.98 %` 寫進 Abstract 錨定 parameter-efficiency claim。

### ☐ Edit X-2 — Chinese 摘要對應修改

**Current:**
> *「在 ACDC 的四種天候條件下都取得一致的 mIoU 提升，可訓練參數只佔 SAM 的極小部分。」*

**After:**
> *「在 ACDC 的四種天候條件下都取得可用的 mIoU，可訓練參數僅佔 SAM 的 2.98%。」*

**Rationale:** 「一致的提升」→「可用的 mIoU」（誠實反映 night 48% 偏低）；同步寫出 `2.98%` 與英文版對齊。

### ☐ Edit X-3 — §1 P3 closing sentence

**Current:**
> *"We validate WeatherSAM on ACDC across fog, rain, snow, and night and find consistent mIoU gains while training only a small fraction of SAM's parameters."*

**After:**
> *"We validate WeatherSAM on ACDC across fog, rain, snow, and night, achieving an overall val mIoU of **65.5 %** while training only **2.98 %** of SAM ViT-H's parameters."*

**Rationale:** 把 placeholder「consistent mIoU gains」換成實測數字 65.5 %（E27 checkpoint）；刪掉「find / gains」因為 per-condition 並不均勻。

### ☐ Edit X-4 — §1 P2 last sentence（motivation gap，optional）

**Current:**
> *"How to inject a confidence-modulated clear-weather reference into a frozen foundation backbone without disturbing its pretrained representation is therefore an open question."*

句子本身沒錯，但既然 parameter-efficiency 已升為主要貢獻，motivation 段應該錨定上去。

**選用建議：** 結尾再加一句——
> *"…is therefore an open question. **A practical answer would also have to update only a fraction of the foundation model's parameters; otherwise the frozen-backbone setup gives away its main advantage.**"*

cheap option，可幫 §4.2 Parameter Efficiency 接力。

### ☐ Edit X-5 — §5 Conclusion P1（results sentence）

**Current:**
> *"On the ACDC benchmark this yields XX.X% overall mIoU across fog, rain, snow, and night while training only X.X% of SAM ViT-H's parameters."*

**After:**
> *"On ACDC validation we obtain **65.5 %** overall mIoU across fog, rain, snow, and night — marginally above Refign-DAFormer's 65.0 % under a UDA-with-reference regime, while remaining 1.7 mIoU below CMA's 67.2 % — and training only **2.98 %** of SAM ViT-H's parameters (**24.5 M of 823.6 M**)."*

**Rationale:**
- 填入實測數字（`65.5` from E27 / `2.98 %` / `24.5 M of 823.6 M`）。
- 把 parity 子句改為更精確的「marginally above Refign / 1.7 below CMA」反映 E27 數字。
- 「ACDC validation」措辭收掉 test-vs-val 的歧義。與 §4.1 的「validation split」措辭對齊。

### ☐ Edit X-6 — §3.2 *Technical advantages*：「preserves」→「is designed to preserve」

self-review 中標記為 `needs evidence` 的「pre-hook placement preserves SAM's residual stream」缺乏 pre vs post-hook ablation 證據。

**Current:**
> *"the pre-hook placement lets the injected reference prior flow through the block's own attention and MLP rather than being added on top of an already-computed output, which **preserves** SAM's pretrained residual stream..."*

**After:**
> *"the pre-hook placement lets the injected reference prior flow through the block's own attention and MLP rather than being added on top of an already-computed output, which **is designed to preserve** SAM's pretrained residual stream..."*

只動一個動詞——把 *claim* 轉成 *design rationale*，移除隱含的實證承諾。

---

# Part C — 最終 §4 章節大綱

```
§4   Experiment
├─ §4.1   Experimental Setup
│         (ACDC val, Cityscapes→ACDC, 19 classes, AdamW,
│          37 epochs trained / model selection at epoch 27, 1024², 24 GB)
├─ §4.2   Parameter Efficiency        ← lead with strength
│         Table 1: per-module breakdown (24.5 M trainable / 823.6 M total)
├─ §4.3   Main Results on ACDC val    ← marginally above Refign, 1.7 below CMA
│         Table 2: overall + per-condition mIoU vs Source / Refign / CMA
├─ §4.4   Per-Class Analysis          ← localise where ref-prior helps and where it doesn't
│         Table 3: 19-class IoU breakdown (highlight rare-dynamic + night-sky drops)
├─ §4.5   Reference Alignment Quality ← visual evidence for the confidence-driven design
│         Fig. 2: warp + confidence overlay across 4 conditions
├─ §4.6   Qualitative Results
│         Fig. 1: WeatherSAM prediction vs GT for 4 sample conditions
└─ §4.7   Limitations and Honest Scoping
          (val-only, regime asymmetry, GNSS pairing requirement)
```

**Note on figures:** 舊草稿中的 *Gate trajectories* 副圖（current Fig. 1 bottom row）應該 **刪除或壓縮成 §4.5 中一句話**。0.050 → 0.058 動態範圍視覺上不夠強。把空間讓給 §4.6 更大的定性比較圖。

---

# Part D — 最終 Claim-Evidence Map

| Claim | Current location | Evidence in revised paper | Status |
|---|---|---|---|
| 2.98 % of SAM is trained | Abstract, §1 P3, §5 | Table 1 (24.5 M / 823.6 M) | ✅ supported |
| **65.5 % overall val mIoU** | Abstract, §1 P3, §5 | Table 2 row WeatherSAM (E27 checkpoint) | ✅ supported |
| **Marginally above Refign-DAFormer (65.0)** | §4.3, §5 | Table 2 + cited Refign Tab 4 | ✅ supported (+0.5 mIoU, not victory) |
| Per-condition split favours static / large vehicles, hurts rare dynamic + night-sky | §4.4 | Table 3 (per-class), Table 2 (per-condition night = 48.85) | ✅ supported |
| Pre-hook placement preserves residual stream | §3.2 *Technical advantages* | **No ablation** | 🟡 weakened to "is designed to preserve" (X-6) |
| Mask2Former-style unified queries beat per-class loop | §3.3 *Technical advantages* | **No ablation** | 🟡 already phrased as "We import...", no claim of superiority |
| Per-token confidence governs reference contribution | §3.2 + Fig. 2 | Fig. 2 (warp + confidence overlay) | ✅ supported qualitatively |
| Reference prior helps most on fog / night | ~~§4.2 Findings~~ | **REMOVED** (no per-condition baseline) | ❌ deleted |
| `+X.X over CMA` | ~~§4.3, §5~~ | CMA val 67.2 > our 65.5 | ❌ deleted (replaced by "1.7 below CMA" honest gap) |

套用 X-1 ~ X-6 + Part C 重排後，論文中所有剩下的 claim 都是 (a) §4 中有數據／視覺證據支持，或 (b) 明確 framing 成 *design rationale* 而非實證 claim。

---

# Part E — Table 1 算術 sanity-check（已解決）

原始稽核時，數字湊不整齊（24.435 vs 24.5、2.967 vs 2.98）。**已用實測 trainable / total 解決：**

```
Trainable:   24,534,329 → 24.5 M
Total:      823,618,244 → 823.6 M
Ratio:    24,534,329 / 823,618,244 = 2.978 %  →  2.98 %
```

→ **Action:** LaTeX commit 時，用以下指令再跑一次取最終 canonical 值，全篇統一捨入到同一位數：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'{trainable:,} / {total:,} = {trainable/total*100:.2f}%')
```

論文 4 處（Abstract、§1 P3、§4.2、§5）都用同一個 `2.98 %`，不要混用 `2.97 %`、`2.98 %`。

---

# Part F — 執行順序

1. **重新跑 parameter count Python snippet**（取得 canonical 24.5 M、823.6 M、2.98 %）
2. **§4.2 ↔ §4.3 對調**：Param Efficiency 在前，Main Results 在後
3. **重編表格**（Part A R2）：Table 1 = param breakdown；Table 2 = main val comparison；Table 3 = per-class
4. **刪掉舊 per-condition Table 4**，數字併入 Table 2 的 WeatherSAM 列
5. **刪掉 / 壓縮 Fig. 1 下半部** gate trajectory（見 Part C note）
6. **套用 X-1 ~ X-6 跨章節編輯**（Abstract、摘要、§1 P3、§1 P2、§5 P1、§3.2 *Tech advantages*）
7. **§4.1 修正 "80 epochs"** → 「trained for 37 epochs (model selection at epoch 27 via val mIoU)」
8. **替換 §4 全段 prose** 為 7 段重寫
9. **Recompile and grep**：
   - `grep -n 'XX.X' main.tex` → 0 hits
   - `grep -n 'consistent mIoU gains' main.tex` → 0 hits
   - `grep -n 'most pronounced on fog and night' main.tex` → 0 hits
   - `grep -n 'over CMA' main.tex` → 0 hits
   - `grep -n '80 epochs' main.tex` → 0 hits
   - `grep -n '64.91' main.tex` → 0 hits（舊 E18 數字）

---

# Summary

| Bucket | Count |
|---|---|
| §4 prose rewrite (7 paragraphs) | 7 |
| Section reordering (§4.2 ↔ §4.3) | 1 |
| Table renumber + merge (4 tables → 3) | 1 |
| Figure compression (drop gate trajectories) | 1 |
| Cross-section claim chase-down (X-1 … X-6) | 6 |
| §4.1 "80 epochs" → "37 epochs / select at 27" | 1 |
| Sweep checks (grep) | 6 |
| Parameter count Python snippet | 1 |
| **Total atomic edits** | **23** |
| Estimated execution time | **~50 min** + 1 LaTeX recompile |

完成後論文具備：
- Zero `XX.X` placeholders
- Zero unsupported 「beats CMA」/「most pronounced on」 claims
- 對 Refign-DAFormer **+0.5 mIoU**（65.5 vs 65.0）與對 CMA **−1.7 mIoU**（65.5 vs 67.2）的誠實陳述，§4.3 與 §5 一致
- **2.98 %** parameter-efficiency 錨點在 Abstract、§1、§4.2、§5 四處同步
- 明確界線的 limitations 段（§4.7），預先處理 reviewer 會問的問題
- 正確的訓練長度陳述：37 epochs（model selection at epoch 27），不是 80

**Framing 核查：** 論文僅宣稱對 Refign-DAFormer **+0.5 mIoU 的微幅領先**（在 UDA validation runs 的雜訊帶內，故措辭為「marginally above」而非「beats」），**不宣稱** 打敗 CMA。獨特賣點為 *"comparable operating point at 2.98 % of the trainable parameter budget"*，這是真實量測且可防禦的。

---

# 後續行動清單（追蹤）

1. ✅ E27 重新評估（完成，commit `020d237`）
2. ✅ Audit plan 更新為 E27 數字（完成，commit `20a1fd0`）
3. ✅ 兩份文件合併為單一權威版本（本檔）
4. ⬜ 把這份數據塞回 `main.tex`（依 Part F 執行順序）
5. ⬜ §4.1 修正「80 epochs」措辭
6. ⬜ （可選）跑 ACDC test set 取得官方 test mIoU
