# 同影像先驗基線（ViT-Adapter 式 SPM）實驗設計

> 日期：2026-08-06 | 分支：`feat/arch-overhaul` | Run ID：`P1_selfprior_seed42`
> 狀態：設計已核准，待實作
> 相關文件：`docs/experiments/2026-07-06-m2f-ablation-plan.md`、`docs/experiments/2026-07-09-ablation-consolidated-report.md`

---

## 1. 背景

口試委員質疑論文直接引用 ACDC 官方基準的 ViT-Adapter 數值（78.4 mIoU）作為比較對象，因其訓練設定不可考，是否適合作為 baseline。

檢視現有實驗後發現，更迫切的問題不在外部引用，而在論文自己的消融表：`W4_seed42`（SAM-Adapter 式同影像注入基線）的 val mIoU 為 79.80，高於完整模型的 76.02 達 3.78 個百分點，卻同時存在三個未對齊的變因：

1. 閘控初始化與排程（W4 固定小值 0.05 起步，Pair-SAM 為 `torch.zeros` 零初始化）
2. 注入位置
3. 有無抽取器

論文 §4.5.2 目前以「零初始化閘控在 30 個週期的預算內尚未追上」解釋此差距。此解釋已被訓練紀錄推翻：

| Run | best epoch | 訓練中 val mIoU 峰值 | gate ep24→30 | gate 前 6 epoch |
|---|---|---|---|---|
| `FULL_seed42` | 22 | 76.10 | 0.01513 → 0.01575 | +0.0055 |
| `W4_seed42` | 18 | 79.79 | 0.07290 → 0.07394 | — |
| `W2_semB_seed42` | 18 | 76.50 | 0.00792 → 0.00828 | — |

> 此表數值取自 `train_log.csv` 的訓練中驗證，與論文所用的正式評估值略有差異，見 §4 註記。

FULL 的最佳點在 ep22，其後八個 epoch 無進步；val 總損失自 ep6 的 27.47 升至 ep30 的 32.94，已進入過擬合；閘控成長率降至前期的約十分之一。兩個 run 均已收斂，差距並非預算不足所致。

因此需要一個與完整模型只差單一變因的受控基線。

## 2. 目標與待答問題

**待答問題：** 在完全相同的主幹、資料、訓練排程與評測協定下，Adapter 的先驗取自「當前影像」而非「跨視角對齊後的晴天參考影像」，效能為何？

**交付物：** 一列可放入論文 ACDC test 比較表的數據，以及對應的 val 逐條件／逐類別結果。

**不在範圍內：** 本次不修改論文任何檔案。數據產出後由使用者決定是否納入。

## 3. 方案選擇

考慮過三種實作，選定方案 B。

| 方案 | 做法 | 判定 |
|---|---|---|
| A | `pre_align(img_curr, img_curr)`，以當前影像作為自己的參考 | 否決。改動最小，但主表格列難以自我解釋（「為何將影像與自身做稠密匹配」），且近似恆等的 flow 會引入翹曲噪聲 |
| **B** | **繞過 UAWarpC，直接以當前影像的 VGG 多尺度特徵為先驗，conf ≡ 1** | **採用** |
| C | 官方 ViT-Adapter 複現（mmsegmentation + BEiT-L） | 否決。另一套 codebase；24GB 單卡須降容量，所得數字既不等於 78.4，也不與 Pair-SAM 同主幹 |

選定 B 的理由：

- **表格列可自我解釋。** 與 RefineNet、HRNet、Mask2Former、ViT-Adapter 並排時，「同影像先驗基線（本文架構，ViT-Adapter 式 SPM）」意義明確。
- **`conf ≡ 1` 不構成第二個變因。** 無翹曲則無對齊不確定性，置信度沒有可調變的對象。設為中性值是 ViT-Adapter 設定的正確實例化，而非移除 Pair-SAM 的組件。
- **直接回應口試質疑。** 回答的是「ViT-Adapter 的機制，在本文的主幹、資料、排程與評測協定下是多少」。

### 命名約束

論文中不得將此列標示為「ViT-Adapter」。本架構的 RPM 取用凍結 VGG 的 l2/l3/l4 特徵，ViT-Adapter 的 SPM 則是從頭訓練的卷積 stem，兩者為功能類比而非等同。建議標示為「同影像先驗基線（本文架構，ViT-Adapter 式 SPM）」並於正文說明差異。

## 4. 實驗矩陣

只新增一列，其餘沿用既有 run，不重跑。

| Run | 先驗來源 | conf | 閘控初始化 | val mIoU | test mIoU |
|---|---|---|---|---|---|
| `FULL_seed42` | UAWarpC 翹曲後之參考影像 | UAWarpC 學習所得 | 零初始化 | 76.02 | 72.14 |
| `W2_semB_seed42` | 全零張量 | ≡ 1 | 零初始化 | 76.50 | — |
| **`P1_selfprior_seed42`** | **當前影像** | **≡ 1** | **零初始化** | 待測 | 待定 |
| `W4_seed42` | 當前影像（SAM-Adapter 變體） | — | 固定 0.05 | 79.80 | — |

> **數值來源一致性（重要）：** 本表 val mIoU 一律取自各 run 的 `e1_results.json` 之 `overall_miou`，即論文所引用的數值。`train_log.csv` 的訓練中驗證峰值採不同評估流程，數值略有出入（FULL 為 76.10 對 76.02，W4 為 79.79 對 79.80）。P1 的結果必須以 `eval_e1_acdc_val_full.py` 產出的 `e1_results.json` 為準，不得以 `train_log.csv` 峰值與本表其他列相比。

P1 與 FULL 之間只有先驗來源一項自由變因（conf 由該來源決定，非獨立選擇）。P1 與 W4 的差異在閘控初始化與注入結構，故 P1 一併回答「W4 的優勢是否源自閘控初始化」。

訓練協定完全沿用 `scripts/ablation_m2f_common.sh` 的 `BASE_FLAGS`：30 epochs、patience 10、batch_size 1、accumulate_steps 4、lr 5e-5、warmup 5、adapter_lr_scale 3.0、label smoothing 0.05、seed 42，起點為 `outputs_m2f_cs_e2e/weather_sam_best_latest.pth`。不覆寫任何既有旗標。

## 5. 實作設計

### 5.1 介面

新增 `train.py` 旗標：

```
--prior_source {reference,self}    預設 reference
```

預設值保持既有行為完全不變。

### 5.2 `fusion.py`：新增 `self_prior()`

鏡像 `CMAAlignment.pre_align()` 的輸出契約，但不執行 UAWarpC：

```
pre_align(curr, ref, l2_native=True)
  → {'l2': warp(VGG_ref.l2), 'l3': warp(VGG_ref.l3), 'l4': warp(VGG_ref.l4), 'mask': conf}

self_prior(curr, out_size, l2_native=True)
  → {'l2': VGG_curr.l2, 'l3': VGG_curr.l3, 'l4': VGG_curr.l4}     # 無 'mask' 鍵
```

實作步驟：呼叫 `self._extract_vgg_features(img_curr)`，取 index 2/3/4，以 `F.interpolate` 調整至與 `pre_align(l2_native=True)` 逐一對等的空間尺寸：

| 鍵 | channel | 空間尺寸 |
|---|---|---|
| `l2` | 256 | `(2*out_H, 2*out_W)` |
| `l3` | 512 | `(out_H, out_W)` |
| `l4` | 512 | `(out_H//2, out_W//2)` |

不設 `'mask'` 鍵。

### 5.3 `deform_adapter.py`：不修改

`ReferencePriorModule.forward()` 以 `feats.get('mask', None)` 取用置信度。省略該鍵時取到 `None`，自動走既有的 else 分支得到 `conf = torch.ones(...)`。此為既有程式碼路徑，不新增分支。

### 5.4 `pair_sam.py`：階段 0 分派

於 forward 的階段 0 依 `self.prior_source` 二選一呼叫 `pre_align()` 或 `self_prior()`。

**約束：** `prior_source='self'` 時不得設定 `_adapter_reference_free`。該旗標供 W4 的 `adapter_variant='sam_adapter'` 使用，會將整個 `vgg_injector` 換掉，與本設計目的不符。

`self` 模式下 `clear_image` 不再是必要輸入，但為維持 dataloader 與其他 run 一致，仍照常載入、僅不使用。

### 5.5 遙測相容

`pair_trainer.py:71-72`、`:629-630`、`:1148-1149` 讀取 `fusion_module._last_conf_mean` 與 `_last_valid_ratio`。`self_prior()` 須將兩者設為 `1.0`，使 `train_log.csv` 的欄位結構與其他 run 一致，`scripts/aggregate_ablation.py` 方能正常彙整。

`_last_flow` 與 `_last_confidence_map` 在 self 模式下無意義，設為 `None`。

### 5.6 執行腳本

新增 `scripts/ablation_m2f_phase6_selfprior.sh`，source `ablation_m2f_common.sh` 並以 `run_one P1_selfprior_seed42 --prior_source self` 呼叫，沿用既有的冪等協定（已有 best ckpt 則跳過訓練，已有 `e1_results.json` 則跳過 eval）。

## 6. 驗證

實作採測試先行。

### 6.1 單元測試

| 檢查 | 判準 |
|---|---|
| shape 對等 | `self_prior()` 三個張量的 shape 與 channel 逐一等同 `pre_align(l2_native=True)` |
| conf 中性 | `ReferencePriorModule.forward()` 回傳的 `conf` 全為 1 |
| 閘控未動 | 各 `Injector.gamma` 初始為零向量，與 FULL 相同 |
| 梯度連通 | 反向傳播後 RPM 的 `proj_c2/c3/c4` 與各 Injector 皆有非零梯度 |
| 遙測欄位 | `_last_conf_mean == 1.0`、`_last_valid_ratio == 1.0` |
| 回歸 | `prior_source='reference'` 的前向輸出與改動前逐位元相同 |

### 6.2 Smoke run

正式訓練前跑 3 個訓練步，確認：

- loss 下降
- `train_inject_gate` 自 0 起始並開始成長
- 峰值顯存低於 24GB（self 模式少一次 VGG 抽特徵並跳過 UAWarpC，應較 FULL 節省）

## 7. 執行流程

```
1. 實作 + 單元測試通過
2. Smoke run（3 步）
3. 訓練 30 epochs, seed 42                    ~6-8 小時
4. eval_e1_acdc_val_full.py                   ~1 小時
   → val 整體 / 逐條件 / 逐類別 mIoU
   ═══ 檢查點：交付數據，由使用者決定是否提交 test ═══
5. dump_acdc_test_preds.py --zip              ~1 小時
6. 上傳 ACDC 官方 evaluation server            提交配額 −1
```

**步驟 4 後停止。** ACDC test 標註不公開，須經官方 server 評分且提交配額不可逆，該步驟由使用者明確指示後才執行。

## 8. 產出

`docs/experiments/2026-08-06-selfprior-baseline.md`，內容包含：

- val 整體 mIoU、逐條件（霧／雨／雪／夜）mIoU、19 類逐類別 IoU
- 訓練曲線與 best epoch
- 與 `FULL_seed42` / `W2_semB_seed42` / `W4_seed42` 的對照表
- `train_inject_gate` 軌跡對比（回答「W4 的優勢是否源自閘控初始化」）
- 若執行至步驟 6：ACDC test 逐類別 IoU 與 mIoU

本次不修改 `paper/` 之下任何檔案。

## 9. 範圍外（YAGNI）

| 項目 | 排除理由 |
|---|---|
| 60 epoch 收斂性檢查 | 既有 log 已證實 FULL best ep22、閘控飽和、val loss 上升，跑更久無助益 |
| 多 seed 重複 | ACDC test 比較表所有列均為單次結果，Pair-SAM 的 72.14 亦然 |
| 方案 A（自我翹曲對照） | 若 P1 結果需要進一步歸因再考慮 |
| 閘控初始化掃描 | 同上，P1 的 gate 軌跡會先給出線索 |
| MUSES 交叉驗證 | 同上 |
| `--prior_source zero` | `W2_semB_seed42` 已涵蓋 |

## 10. 預期風險

**P1 可能高於 Pair-SAM。** `W4_seed42` 的 79.80 若反映的是注入內容而非閘控差異，P1 的 val 可能落在 79 附近，對應 test 約 76，高於 Pair-SAM 的 72.14。

此為誠實的實驗結果。屆時的處理選項（由使用者決定）：

1. 納入主比較表，論述改為「跨視角參考的價值在特定條件而非整體平均」，以既有的逐條件數據（雨 +1.55、雪 +2.09、霧 −2.41、夜 −1.71）支撐
2. 置於消融章節而非主比較表
3. 不納入，僅作為內部驗證

選項 3 仍應在論文中修正 §4.5.2 對 W4 的收斂性解釋，因該解釋已被訓練紀錄推翻。
