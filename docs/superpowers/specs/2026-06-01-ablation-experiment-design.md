# 消融實驗框架設計 spec（第四章 4.9 節）

> 對應論文 `chapter4-experiment.tex` 第 4.9 節。本 spec 依「核對現有程式碼 → brainstorming 決策」整理而成，作為 writing-plans 拆解實作計畫的依據。
> 撰寫日期：2026-06-01。

---

## 0. 已定案決策（brainstorming 結果）

| 決策 | 選定 |
|------|------|
| 表格範圍 | **3 張表 = 累積表 `tab:ablation_summary` + adapter 表 `tab:adapter_ablation` + loss 表 `tab:loss_ablation`；共 10 個 unique configs** |
| 刪除的論文內容 | **僅刪 4.9.2 decoder 表**（`tab:decoder_ablation`）；保留 4.9.1 adapter、4.9.3 loss 兩表 |
| FULL 來源 | **全部從 SAM checkpoint 重訓**，同一訓練資本（不沿用 best E27 權重） |
| seed 策略 | **R1、FULL、A2 各跑 3 seeds 報 mean±std**；R2–R6、A1、C1 各單 seed |
| A2 移除 reference 語意 | **零張量取代 reference 特徵**（保 adapter 參數量不變，精確隔離「資訊 vs 容量」） |
| 補充指標 | **只加 per-class IoU**（rider/moto/bike，eval 已免費輸出）；**不實作 Boundary metric** |
| 開關實作風格 | **延伸現有 argparse + `run_ablation.sh` 釘住每條指令**（不引入 YAML config 系統） |

### 表格範圍的取捨依據

- **累積表**（R1–R6+FULL）：逐次加入單一模組，觀察邊際貢獻。
- **adapter 表**（FULL/A1 後置注入/A2 不引入 reference）：A2 是唯一能分離「參考資訊 vs adapter 容量」的控制組（無累積表對應）；A1 提供完整基準下的 post-vs-pre leave-one-out。
- **loss 表**（FULL/C1 純CE/C2 取消MFB）：以表格呈現 rider/moto/bike 長尾 per-class IoU，呼應 §4.5；C2 與 R6 為**同一 config**（免費複用）。
- **decoder 表刪除**：B1（逐類別）↔ 累積 R3→R4、B2（無LRH）↔ R4→R5 已涵蓋，且 B2 因不做 Boundary metric 而證據單薄，論述改折進累積表。
- **救回 adapter+loss 表不需新增程式開關**：A1 用 `--inject post`、C1 用 `--lovasz_weight 0 --dice_weight 0`，皆既有/已規劃開關，僅各多跑 1 run。

---

## 1. Run 矩陣（10 個 unique configs）

完整模型 **FULL = 現行 `train.py` 預設 config**：adapter on、inject=pre、decoder=unified、lrh=on、loss=CE+Lovász+Dice、mfb=on。

**累積表 R1–R6 + FULL**（每列相對前一列只改一維度）：

| Run | use_vgg_adapter | inject | decoder | lrh | loss(lovasz/dice) | mfb | 相對前列新增 | seeds |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|---|:--:|
| **R1** baseline | **off** | – | per-class | off | CE only (0/0) | off | 裸 SAM + 逐類別 + 純CE | **3** |
| **R2** | on | **post** | per-class | off | CE only | off | +Ref（後置注入） | 1 |
| **R3** | on | **pre** | per-class | off | CE only | off | 後置→前置 | 1 |
| **R4** | on | pre | **unified** | off | CE only | off | 逐類別→統一查詢 | 1 |
| **R5** | on | pre | unified | **on** | CE only | off | +LRH | 1 |
| **R6** | on | pre | unified | on | **CE+Lov+Dice** | off | +Lovász/Dice | 1 |
| **FULL** (=列7) | on | pre | unified | on | CE+Lov+Dice | **on** | +MFB | **3** |

**leave-one-out 變體**（= FULL 改單一維度，供 adapter / loss 表）：

| Run | 用於表 | = FULL 但… | inject | loss(lovasz/dice) | mfb | seeds |
|-----|------|-----------|:--:|:--:|:--:|:--:|
| **A1** | adapter 表 | 後置注入 | **post** | CE+Lov+Dice | on | 1 |
| **A2** | adapter 表 | 移除 reference（零張量） | pre | CE+Lov+Dice | on | **3** |
| **C1** | loss 表 | 純 CE | pre | **CE only (0/0)** | on | 1 |
| **C2** | loss 表 | 取消 MFB | pre | CE+Lov+Dice | **off** | — (= R6) |

> A2 額外帶 `--ref off`（其餘 = FULL）。C2 config 與 R6 完全相同，**直接複用 R6 的 checkpoint/metrics，不另訓**。

**unique configs**：R1, R2, R3, R4, R5, R6(=C2), FULL, A1, A2, C1 = **10**。
**訓練次數合計**：R1(3) + R2–R6(5) + FULL(3) + A2(3) + A1(1) + C1(1) = **16 次訓練**。

---

## 2. 需實作的程式開關（5 處）

所有開關以 argparse flag 暴露，預設值 = FULL 設定（向後相容，不改變現行訓練行為）。

### 2.1 `--inject {pre,post}` — 🟢 接線（模型已支援）
- **現況**：[weather_sam.py:93](../../segment-anything/segment_anything/modeling/weather_sam.py) `enable_vgg_adapter(mode='pre'/'post')` 已實作；[train.py:258](../../segment-anything/train.py) 寫死呼叫預設 `'pre'`。
- **改動**：train.py 加 `--inject`（default `pre`），把 `model.enable_vgg_adapter(mode=args.inject)`。
- **觸及**：train.py 單處。

### 2.2 `--decoder {unified,per_class}` — 🟡 新增 forward 路徑（參數量不變）
- **現況**：[weather_mask_decoder.py:96](../../segment-anything/segment_anything/modeling/weather_mask_decoder.py) `predict_masks_semantic` 將 K 個 class query 放同一 transformer sequence（跨類別 self-attention）。
- **改動**：新增 `predict_masks_per_class`：對每個 class_id 用 `[單一 class query + 該類 prompt token]` 單獨呼叫 transformer K 次（K=1），**複用同一組 `class_mask_tokens` / `class_hypernetworks_mlps` / transformer**，移除跨類別 self-attention。以 `self.decoder_mode` 屬性切換，`forward_semantic` 依此分派。
- **參數可比性**：per-class 與 unified 共用完全相同的模組與權重，**參數量相同**，僅推論流程（K 次 vs 1 次 forward）不同 → 滿足論文可比性要求。
- **觸及**：weather_mask_decoder.py（新方法 + 分派）、weather_sam.py（傳遞 mode）、train.py（flag）。

### 2.3 `--lrh {on,off}` — 🟡 gate（三處一致）
- **現況**：`context_fusion_head`（LRH）**不在** model.forward 內，而在外部套用：trainer [weather_trainer.py:429](../../segment-anything/weather_trainer.py)（train）、validate 段、eval [eval_e1_acdc_val_full.py:68](../../segment-anything/scripts/eval/eval_e1_acdc_val_full.py)。
- **改動**：以 `model.use_lrh`（bool）為單一真值來源。三處統一：`logits = model.context_fusion_head(full) if model.use_lrh else full`。
- **觸及**：weather_sam.py（屬性）、weather_trainer.py（train + validate 兩處）、eval 腳本。**一致性是此開關的最大陷阱，需測試保證 train/eval 行為一致。**

### 2.4 `--mfb {on,off}` — 🟢 權重換 uniform
- **現況**：CE 恆用 MFB 權重（[new_loss.py:132-135](../../segment-anything/utils/new_loss.py) `class_weights = _build_median_freq_weights(...)`）；mask dice 加權用 `_mask_cls_w = ACDC_CLASS_WEIGHTS`（[weather_trainer.py:117](../../segment-anything/weather_trainer.py)）。
- **改動**：`ContextLoss` 加 `use_mfb` 參數；off 時 CE 的 `weight=None`（uniform）。trainer 的 `_mask_cls_w` off 時設為 `ones(19)`。
- **觸及**：new_loss.py（ContextLoss）、weather_trainer.py（兩處 cls_w 來源）、train.py（flag）。

### 2.5 `--ref {on,off}` — 🟡 移除 reference 資訊（A2 專用）
- **現況**：[vgg_adapter.py:179-180](../../segment-anything/segment_anything/modeling/vgg_adapter.py) K/V 來自 reference VGG 特徵 `f_flat`。
- **改動（已定案）**：`--ref off` 時，將餵入 `k_projs/v_projs` 的 reference 特徵以**零張量**取代（`f_flat → zeros_like(f_flat)`），使 attention 無參考內容可 attend、退化為注入一個學習到的常數補償。
  - **理由**：此法**保持 adapter 參數量與結構完全不變**，只移除「參考內容」，精確隔離「reference 資訊 vs adapter 容量」—— 正是 A2 的科學目的。
- **觸及**：vgg_adapter.py（`_inject_at_stage` 加 ref 開關）、weather_sam.py（傳遞）、train.py（flag）。

> **R1 不需 `--ref`**：R1 用既有 `--no-use_vgg_adapter` 整個關閉 adapter（走 cache 路徑），與 A2「保留 adapter 但無 reference」語意不同。

### 2.6 既有 flag（零改動）
- `--use_vgg_adapter` / `--no-use_vgg_adapter`：R1 用。
- `--lovasz_weight 0 --dice_weight 0`：退化純 CE（R1–R5）。

---

## 3. Config 一致性機制（學術可信度核心）

**問題**：eval 端目前寫死 ckpt 路徑、寫死「套用 LRH」、寫死 decoder 行為（[_eval_common.py:70](../../segment-anything/scripts/eval/_eval_common.py) `load_weather_sam_model`）。若每個 run 的 decoder mode / lrh / ref 不同，eval 必須以**完全相同的 config** 重建模型，否則 train/eval 不一致 → 數據無效。

**機制**：
1. **訓練落地 config**：train.py 在 `output_dir` 寫 `ablation_config.json`，記錄該 run 的 `{inject, decoder, lrh, mfb, ref, use_vgg_adapter, lovasz_weight, dice_weight, seed}`。
2. **eval 讀 config**：eval 腳本加 `--ckpt` 與 `--config`（或自動找 ckpt 同目錄的 json），據此重建模型、決定是否套 LRH、選 decoder mode。
3. **杜絕硬編碼**：`load_weather_sam_model` 改為接受 config dict。

---

## 4. 評估與彙整（評估端零新指標，全為重用 + 彙整）

1. **per-condition / per-class / overall mIoU**：[eval_e1_acdc_val_full.py](../../segment-anything/scripts/eval/eval_e1_acdc_val_full.py) 已輸出 JSON（per-class×per-condition IoU 矩陣 + per-condition mIoU + overall）。改為 ckpt/config-aware 後逐 run 跑。
2. **`aggregate_ablation.py`（新）**：掃 10 個 unique config 的 JSON，輸出 3 張表的 `.tex` 可貼片段：
   - **`tab:ablation_summary`**：7 列（R1–R6+FULL）All mIoU + Δ，可選 per-condition；R1、FULL 標 mean±std（3-seed）。
   - **`tab:adapter_ablation`**：FULL / A1（後置注入）/ A2（不引入 reference）× Fog/Rain/Snow/Night/All/Δ；FULL、A2 標 mean±std。
   - **`tab:loss_ablation`**：FULL / C1（純CE）/ C2=R6（取消MFB）× All mIoU + rider/moto/bike IoU + Δ；FULL 標 mean±std。
   - 對 3-seed runs（R1、FULL、A2）計算 mean±std。
3. **ACDC 類別像素頻率**：已內建於 [new_loss.py:5](../../segment-anything/utils/new_loss.py) `_ACDC_CLASS_FREQ`（1200 張實測），rider/moto/bike 比例直接導出，填 4.9.3 正文（`[X.XXX]%`）。

---

## 5. 驗證策略（確保符合學術標準）

### 5.1 開關正確性（TDD 單元測試）
每個開關需測試「確實改變 forward 行為」：
- `--decoder`：(a) per-class 與 unified **參數量相同**；(b) K=1 時兩路徑數值輸出相同；(c) K>1 時 per-class 無跨類別交互（改一類 query 不影響他類輸出）。
- `--lrh off`：輸出 == 未經 context_fusion_head 的 assembled logits；train 與 eval 路徑一致。
- `--mfb off`：ContextLoss 的 CE 權重為 uniform（等同無 weight）。
- `--ref off`：injector 對 reference 內容不敏感（換不同 reference 影像，輸出不變）；參數量與 `--ref on` 相同。
- `--inject post`：hook 註冊在 forward hook（後置）而非 pre_hook。

### 5.2 整合 smoke 測試
- 10 個 unique config 各跑 **0–1 epoch smoke**，確認：能跑通、`ablation_config.json` 正確落地、eval 能據 config 重建並算出數字。

### 5.3 FULL 重訓 pipeline sanity gate
- FULL = 現行預設 config，重訓後 val mIoU 應落在 **best E27（65.68）同量級**（容許 seed 造成的 ±~0.5）。若顯著偏低，代表 pipeline 退化，須先除錯再跑其餘 runs。**此為執行順序第一道關卡。**

### 5.4 數據誠實性
- 嚴守論文原文 4.10 的誠實檢討框架：實際 Δ 若與預期方向相左（例如 LRH/MFB 對 overall mIoU 增益極小或為負），**修改正文敘述**而非調整數據。累積表的「逐列遞增」是預期假設，非保證。

---

## 6. 執行順序與算力

**單 run 估**：最多 80 epoch × 1600 張，單張 RTX 4090，依 early stopping 實際約半天～一天。

**建議順序（由風險高/資訊量大優先）**：
1. **FULL**（3 seeds）— pipeline sanity gate，先確認可重現 E27 量級。
2. **A2**（3 seeds）與 **R1**（3 seeds）— 驗證兩端點，A2 驗證中心論點，早期發現 pipeline 問題。
3. R2–R6 累積中間列。
4. A1（後置注入）、C1（純CE）— 補齊 adapter / loss 表的 leave-one-out 變體。
5. 全跑完 → `aggregate_ablation.py` 產出 3 張表 `.tex` + 正文數值。

**並行**：16 次訓練彼此獨立，若有第二張卡可平行。

---

## 7. 交付物

1. **程式**：5 個 argparse 開關（含 2.1–2.5 觸及檔案的外科式修改）+ `ablation_config.json` 落地 + eval ckpt/config-aware 改造 + `aggregate_ablation.py`。
2. **測試**：5.1 的開關單元測試 + 5.2 smoke。
3. **腳本**：`run_ablation.sh`（16 條訓練指令 + eval + 彙整，釘死每 run 的 flag 與 seed，可重現）。
4. **數據產物**：10 個 unique config 的 metrics JSON + 3 張表（`tab:ablation_summary` / `tab:adapter_ablation` / `tab:loss_ablation`）的 `.tex` 可貼片段 + 正文待填數值（rider/moto/bike IoU、類別像素頻率）。
5. **論文改寫**：見獨立文件 [`2026-06-01-paper-rewrite-4.9-ablation.md`](2026-06-01-paper-rewrite-4.9-ablation.md)（刪 4.9.1–4.9.3 三表、A2 折進論述、交叉引用搶救等），不直接改 .tex。

---

## 8.5 模組化評估與決策

**證據**：「組裝 19 類 logits（`-10.0` scatter）+ 套用 `context_fusion_head`」邏輯在 7 處重複——trainer train(426)、trainer validate(932)、eval_e1_full(62)、eval_paper(80)、dump_test_preds(159)、viz_e4(58)、test_inference(55)。`--lrh` 若不模組化須在 7 處各加條件判斷，極易造成 train/eval 套用不一致（spec §3 最大風險）。

**納入計畫的模組化（僅服務開關，不做 trainer 拆分等臆測性重構）**：

| 項 | 內容 | 服務 | 取捨 |
|---|---|---|---|
| **A** | `assemble_semantic_logits(model, low_res, class_ids, *, use_lrh)` 共用函式：單一來源做 scatter + gated LRH；遷移 trainer(train/validate) 與 eval_e1_full 三個**消融路徑**呼叫點 | `--lrh` 單點開關 + train/eval 一致 | **納入（必要）** |
| **B** | `build_weather_sam_from_config(cfg, ckpt)`：單一模型建構路徑吃 5 開關，train.py 與消融 eval 共用，取代寫死的 `load_weather_sam_model` | config 一致性（§3） | **納入（必要）** |
| **C** | `ablation_config.json` 落地/讀取 glue（非 YAML 框架，輕量 dataclass 或 dict） | 5 開關 threading 不出錯 | **納入（輕量）** |
| **D** | 遷移 #4–#7（eval_paper / dump_test_preds / viz_e4 / test_inference）至共用函式 | 一致性清理 | **可選 follow-up，與主線分離，不阻塞** |

> A 的共用函式置於合適模組（如 `segment_anything/modeling/` 或 `utils/`），由 writing-plans 定檔名。**不**重構 1061 行 trainer 結構、不動無關程式。

---

## 8. 開放項（已全部結案）

1. **A2 seeds** → ✅ **3 seeds**（報 mean±std）。
2. **A2「移除 reference」語意** → ✅ **零張量取代 reference 特徵**（保 adapter 參數量不變）。
3. **實作風格** → ✅ **argparse + `run_ablation.sh`**。
4. **表格範圍 / 論文改寫** → ✅ **保留 adapter + loss 表，僅刪 decoder 表（4.9.2）**；改寫細節見 [`2026-06-01-paper-rewrite-4.9-ablation.md`](2026-06-01-paper-rewrite-4.9-ablation.md)。
