# Rare Class Sampling (RCS) 設計 spec

> 將 DAFormer 的 Rare Class Sampling 導入 WeatherSAM 監督訓練，並納入第 4.9 節消融框架。
> 對應並修訂 `2026-06-01-ablation-experiment-design.md`、`2026-06-01-paper-rewrite-4.9-ablation.md`。
> 撰寫日期：2026-06-03。

---

## 0. 動機與定案決策

**動機**：消融診斷顯示 FULL（不含 RCS）在長尾類（bus/motorcycle/bicycle）IoU 偏低且逐 seed 高變異；E27 的 65.68% 主要是單一 seed 在這些類別的有利抽樣。RCS（DAFormer, Hoyer et al. CVPR 2022）對「含稀有類的影像」過取樣，直接提升稀有類 IoU 並降低變異，是對症的方法改良（與既有 loss 端 MFB 互補）。

| 決策 | 選定 |
|------|------|
| 取樣忠實度 | **忠實 DAFormer**：每步先依 `P(c)` 抽類別、再從含該類影像均勻抽一張 |
| 消融定位 | RCS 併入 FULL 成為新完整模型；累積表**最後一步 R8** = `+RCS` = 新 FULL |
| RCS leave-one-out | = R7（FULL 去 RCS），即累積表第 7 列，**免額外 run** |
| 影響範圍 | **僅訓練時資料取樣**；不改模型 → eval / `build_weather_sam_from_config` 不受影響 |
| 溫度 | `T = 0.01`（DAFormer 預設） |
| seeds | **僅 FULL(R8) 跑 3 seeds**；其餘 11 個 config 各 1 seed |

---

## 1. 演算法（忠實 DAFormer Eq.7）

**類別取樣機率**（依像素頻率，rare class 機率高）：
```
P(c) = exp((1 - f(c)) / T) / Σ_c' exp((1 - f(c')) / T)
```
- `f(c)`：類別 c 的像素頻率（正規化到 [0,1]），用既有 `utils/new_loss.py:_ACDC_CLASS_FREQ`（1200 張實測；本 spec 沿用，並於 §3 驗證其與 1600-train 一致性，必要時以 precompute 的像素計數重算）。
- `T = 0.01`。

**每個訓練樣本的抽法**：
1. 依 `P(c)` 抽一個類別 `c`。
2. 從「GT 中含類別 c」的影像清單中均勻隨機抽一張，回傳其 dataset index。

一個 epoch 產生 `len(dataset)` 個 index（replacement 抽樣）。所有隨機性由綁定 seed 的 `torch.Generator` 驅動 → 可重現。

> 忽略類別 255 不參與 P(c)。某類別若無任何影像含之（理論上不會），P(c) 該項設 0 並重正規化。

---

## 2. 元件與檔案

### 2.1 每影像類別表（precompute）— 新增 `scripts/precompute_class_presence.py`
- 讀 train CSV 的 `gt_path` 欄，逐張 `cv2.imread(gt, GRAYSCALE)`，記錄出現的類別 id（0–18；排除 255）。
- 輸出快取 `class_presence.json`：`{ gt_path: [class_ids], ... }` + 全域 `class_pixel_counts`（供必要時重算 f(c)）。
- 一次性（~1600 張）；輸出存於 train CSV 同目錄或指定 `--out`。
- 冪等：若快取存在且較 CSV 新則略過（除非 `--force`）。

### 2.2 取樣器 — 新增 `utils/rare_class_sampler.py::RareClassSampler(torch.utils.data.Sampler)`
- `__init__(class_presence, class_freq, num_samples, temperature=0.01, seed=42)`：
  - 建 `P(c)` 向量；建 `class_to_indices`（dict: c → list[dataset index]，由 class_presence 反轉）。
  - 綁 `torch.Generator().manual_seed(seed)`。
- `__iter__`：產生 `num_samples` 個 index —— 每個 = 抽 c~P(c)、再從 `class_to_indices[c]` 均勻抽。
- `__len__ = num_samples`。
- 與 DataLoader `shuffle` 互斥（用 sampler 時不可設 shuffle）。

### 2.3 train.py 接線
- flag：`--rcs`（`BooleanOptionalAction`, default **True**）、`--rcs_temp`（float, default 0.01）、`--class_presence`（path，default 取 train_csv 同目錄 `class_presence.json`）。
- `--rcs` on：建 `RareClassSampler(seed=args.seed)`，train_loader 用 `sampler=`（移除 `shuffle=True`）；off：維持 `shuffle=True`（現狀）。
- val_loader 不變（RCS 只作用於訓練）。
- `ablation_config.json` 增記 `rcs`、`rcs_temp`（provenance；eval 不需）。

### 2.4 eval / 模型建構
- **無需改動**。RCS 不在模型內、不留任何 state；`build_weather_sam_from_config` 與 eval_e1 不讀 rcs。

---

## 3. 消融框架修訂（連動 `2026-06-01-ablation-experiment-design.md`）

**累積序列（12 configs，RCS 為最後一步）**：

| Run | 相對前列新增 | rcs | 備註 |
|-----|------|:--:|---|
| R1 | 裸 SAM 基線 | off | |
| R2 | +Ref（後置注入） | off | |
| R3 | +前置注入 | off | |
| R4 | +統一查詢 | off | |
| R5 | +LRH | off | |
| R6 | +Lovász/Dice | off | |
| R7 | +MFB | off | **舊 FULL；= 新 FULL 去 RCS（RCS 控制組）** |
| **R8 = FULL** | **+RCS** | **on** | **新完整模型、論文主數字** |

**leave-one-out（皆相對新 FULL，rcs on）**：A1（後置注入）、A2（移除 reference）、C1（純 CE）、C2（取消 MFB）。
> ⚠️ FULL 含 RCS 後，**C2 ≠ R6**（C2 = mfb off + rcs on；R6 = mfb off + rcs off）→ C2 為獨立 run（原本可複用 R6 的捷徑失效）。RCS 控制組則複用 R7。

**unique configs = 12**：R1–R8、A1、A2、C1、C2。
**seeds**：FULL(R8) ×3；其餘 ×1 → **訓練 14 次**。

---

## 4. 文件連動更新清單

| 文件 | 更新內容 |
|------|---------|
| `2026-06-01-ablation-experiment-design.md` | run 矩陣改 12 configs / 14 runs；新增 R8/RCS 維度；seeds 改「僅 FULL ×3」；C2 不再複用 R6 之註記；新增 `--rcs`/`--rcs_temp` 開關（標註「train-only、eval 不受影響」） |
| `run_ablation.sh` | 全 run 加對應 `--rcs/--no-rcs`；R1–R7、A1/A2/C1/C2 用 `--no-rcs`，R8(FULL) 用 `--rcs`；新增 R7 與 R8 兩列；seeds 僅 FULL ×3；先跑 precompute |
| `ABLATION_RUNBOOK.md` | Phase 0 加「precompute class_presence」；Phase 2 FULL 指令加 `--rcs`；run 矩陣/順序更新 |
| `2026-06-01-paper-rewrite-4.9-ablation.md` | 累積表 +1 列（R8 +RCS）；新增 RCS 方法敘述（P(c)、T、先抽類別再抽影像、可重現性）；長尾證據改以 R7→R8 佐證 RCS |
| `aggregate_ablation.py` | 累積表支援 R8 列（R1–R8）；FULL 指向 R8 |

---

## 5. 驗證策略（學術標準）

### 5.1 單元測試（`tests/test_rare_class_sampler.py`）
- **機率正確性**：固定 class_presence + freq，抽 50k 樣本，統計「被抽類別」分布 ≈ `P(c)`（卡方/相對誤差容忍）。
- **稀有類過取樣**：稀有類（bus/moto/bike）對應影像的被抽比例**顯著高於** uniform shuffle（呼應 DAFormer Fig. S1）。
- **可重現性**：同 seed 兩次 `__iter__` 產生相同 index 序列；不同 seed 不同。
- **覆蓋性**：每個非空類別都可能被抽到；index 皆 < num_samples。
- **precompute 正確性**：對小型合成遮罩，class_presence 結果正確（含 255 排除）。

### 5.2 整合 smoke
- `--rcs` on 跑 1 epoch：sampler 正常產 index、訓練不崩、config.json 記 rcs=true。
- 對照 `--no-rcs` 1 epoch：行為同現狀。

### 5.3 效果驗證（跑完後）
- R7(no-RCS) vs R8(RCS)：預期 **bus/moto/bicycle IoU 上升**、且 FULL 3-seed 的稀有類 std 應較小（RCS 穩定長尾）。誠實標註：若 overall mIoU 增益有限亦如實陳述。

---

## 6. 風險與注意

- **f(c) 來源一致性**：`_ACDC_CLASS_FREQ` 為 1200 張版本；precompute 會輸出 1600-train 的像素計數，spec §1 以 precompute 版為準（或驗證兩者差異可忽略）。實作時於 spec/正文交代採用哪一版。
- **與 MFB 不重複計算**：RCS（資料端）與 MFB（loss 端）是兩個獨立平衡機制，可並存；論文須分開描述。
- **可重現性**：sampler 的 generator 必須由 `args.seed` 衍生，且 DataLoader `worker_init_fn` 既有 seed 機制不衝突（sampler 在主程序產 index，worker 只做資料讀取）。
- **不可同時 shuffle+sampler**：on 時務必移除 `shuffle=True`。

---

## 7. 開放項（已結案）
1. **f(c) 來源** → ✅ **precompute 重算（1600-train 真實像素計數）**。precompute 輸出 `class_pixel_counts`，RareClassSampler 由此算 f(c) 與 P(c)（不沿用 1200 版 `_ACDC_CLASS_FREQ`）。
2. **train.py `--rcs` 預設** → ✅ **預設 on**（RCS 為新完整模型行為）。消融指令對 R1–R7、A1/A2/C1/C2 顯式帶 `--no-rcs`；R8(FULL) 用預設 on（或顯式 `--rcs`）。
