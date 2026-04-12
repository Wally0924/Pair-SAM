# WeatherSAM 研究報告書
**版本**：v1.0　｜　**日期**：2026-04-11　｜　**作者**：WeatherSAM Research

---

## 目錄

1. [專案概述](#1-專案概述)
2. [現有架構說明](#2-現有架構說明)
3. [已完成的修改](#3-已完成的修改)
4. [待執行的架構重構](#4-待執行的架構重構)
5. [核心研究方向：Location-Aware Contrastive Loss](#5-核心研究方向location-aware-contrastive-loss)
6. [可解釋性驗證計畫](#6-可解釋性驗證計畫)
7. [實驗設計與消融研究](#7-實驗設計與消融研究)
8. [相關文獻方向](#8-相關文獻方向)
9. [執行優先順序](#9-執行優先順序)

---

## 1. 專案概述

### 1.1 研究目標

WeatherSAM 是一個基於 Segment Anything Model（SAM ViT-H）的惡劣天氣語意分割模型。核心研究命題為：

> **同地點的晴天影像包含能夠幫助惡劣天氣語意分割的先驗資訊，且此對應關係可被模型顯式學習與驗證。**

### 1.2 資料集

| 資料集 | 用途 | 備註 |
|---|---|---|
| Cityscapes | 現有訓練基線 | `train_with_gps.csv` |
| ACDC | 惡劣天氣訓練目標 | fog / rain / snow，無 night |

### 1.3 ACDC CSV 格式（`Datasets/ref_complete.csv`）

| 欄位 | 內容 | 用途 |
|---|---|---|
| `image_path` | 惡劣天氣 RGB（`_rgb_anon.png`） | ViT-H 輸入 |
| `ref_mask_path` | 晴天 color label（`_gt_ref_labelColor.png`） | MaskEncoder 輸入（現況）→ 未來改為晴天 RGB |
| `gt_path` | 惡劣天氣 labelTrainIds（`_gt_labelTrainIds.png`） | CrossEntropyLoss target |
| `lat` / `lon` | 填 0.0（ACDC 無 GPS） | LocationEncoder 移除後整欄廢棄 |
| `invalid_mask` | ACDC 官方無效區域遮罩 | 移除固定盲區（車頭、相機遮擋） |

驗證結果：750 筆資料全數通過格式驗證（2026-04-11）。

---

## 2. 現有架構說明

### 2.1 資料流

```
[輸入]
  adverse RGB (1024×1024)
  clear color label (ref_mask_path)
  text prompts (class names)
  GPS coords (lat, lon)

[前向傳播]
  adverse RGB
    → preprocess (normalize + pad)
    → ViT-H ImageEncoder
    → image_embedding (B, 256, 64, 64)

  clear color label
    → MaskEncoder (4-layer CNN)
    → ref_embedding (B, 256, 64, 64)

  image_embedding + ref_embedding
    → CrossViewAlignment (cross-attention)
    → aligned_embedding (B, 256, 64, 64)

  image_embedding + aligned_embedding
    → GatedFusion (gate net + LayerNorm)
    → fused_embedding (B, 256, 64, 64)

  text prompts
    → TextEncoder (CLIP projection)
    → sparse_embeddings (K, 1, 256)

  GPS coords
    → LocationEncoder
    → location_embeddings (K, 1, 256)

  sparse_embeddings + location_embeddings
    → WeatherPromptEncoder (concat → K, 2, 256)

  fused_embedding + prompt_embeddings
    → MaskDecoder (Transformer)
    → low_res_logits (K, 3, 256, 256)
    → iou_predictions (K, 3)

  low_res_logits (best candidate)
    → ContextFusionHead
    → fused_logits_hr (1, 19, 1024, 1024)

[損失函數]
  MaskLoss    = Focal Loss + Dice Loss  （binary，per class）
  IoU MSE     = MSE(iou_pred, true_iou)
  ContextLoss = CrossEntropyLoss(ignore_index=255)
  ABL         = Active Boundary Loss（Epoch ≥ abl_start_epoch 後啟動）
```

### 2.2 Ignore Index 處理機制

無效區域（pixel 值 = 255）在所有 loss 中的排除邏輯：

| Loss | 排除機制 |
|---|---|
| CrossEntropyLoss | `ignore_index=255`，PyTorch 原生支援 |
| Focal + Dice | `valid_mask = (gt_mask != 255).float()` 乘法遮罩 |
| ABL | `valid_region = (gt != 255)` 過濾 KL map 及邊界提取 |

**ACDC invalid_mask 的合流邏輯（已實作）**：

```
入口 1：gt_path 原生的 255（標註缺失）
入口 2：invalid_mask（車頭遮擋等固定盲區）→ trainer 中強制設為 255
              ↓
        統一以 ignore_index=255 排除，三個 loss 一次處理
```

---

## 3. 已完成的修改

### 3.1 ACDC Invalid Mask 支援（2026-04-11）

**`segment-anything/utils/weather_dataloader.py`**

- `__getitem__`：新增 `invalid_mask` 讀取邏輯。
  - 若 CSV 有 `invalid_mask` 欄位且檔案存在 → 讀取 PNG，`inv == 0` 為 True（無效）
  - 否則 → 輸出全 False tensor，維持與 Cityscapes 相同介面
- `collate_fn`：新增 `invalid_masks` stack，`batch_dict` 新增 `"invalid_mask"` key

**`segment-anything/weather_trainer.py`**

- `train_epoch` 與 `validate_epoch` 的 `valid_mask_i` 計算前，加入：
  ```python
  if batch['invalid_mask'][i].any():
      gt_mask_i = gt_mask_i.clone()
      gt_mask_i[batch['invalid_mask'][i].to(self.device).unsqueeze(0)] = 255
  ```
- 兩個資料集的行為完全一致：Cityscapes 的 `invalid_mask` 全為 False，此判斷短路，零效能損耗。

### 3.2 資料驗證工具

- **`validate_acdc_csv.py`**：驗證 CSV 中每筆資料的三個欄位格式正確性
  - `image_path`：3-channel RGB 可讀
  - `ref_mask_path`：3-channel color PNG，非全黑
  - `gt_path`：pixel 值 ∈ {0..18, 255}
  - 結果：750/750 通過

- **`visualize_acdc_sample.py`**：隨機抽樣視覺化，三欄並排（adverse RGB / clear color label / GT colorized）

---

## 4. 待執行的架構重構

> 詳細修改位置見：`REFACTOR_PLAN_remove_location_rgb_ref.md`

### 4.1 移除 LocationEncoder

**動機**：LocationEncoder 以 GPS 座標作為地點表示，在 ACDC 750 筆資料量下，座標→場景的映射無法有效學習，實驗確認對 mIoU 無顯著貢獻。地點資訊應以視覺形式（晴天 RGB）直接輸入，而非符號形式（座標）。

**影響範圍**：

| 檔案 | 修改內容 |
|---|---|
| `weather_sam.py` | 移除 `location_encoder` 參數、移除 forward 中的 Location Encoding 區塊、`location_embeddings` 改傳 `None` |
| `build_weather_sam.py` | 移除 `LocationEncoder` import 與初始化 |
| `weather_dataloader.py` | 移除 GPS 讀取與加噪邏輯、移除 `output["location"]`、`collate_fn` 移除 `location` |
| `weather_trainer.py` | `_prepare_batch_input` 移除 `'location'` key |
| `location_encoder.py` | 整個封存（確認穩定後刪除） |

**不受影響**：`WeatherPromptEncoder`（`location_embeddings=None` 已是 optional）、`MaskDecoder`（sparse token 從 K×2 變回 K×1 完全相容）。

**注意**：舊 checkpoint 含 `location_encoder.*` key，重新載入需用 `strict=False`。

### 4.2 Ref 輸入從 Color Label 改為晴天 RGB

**動機**：Color label 將所有語義資訊壓縮為調色盤顏色，MaskEncoder 只能提取語義邊界，失去紋理與外觀資訊。改用晴天 RGB 後，CrossViewAlignment 從「語義對齊」升級為「外觀對齊」，讓模型明確知道「這個地點在晴天長什麼樣子」。

**架構資料流變化**：

```
修改前：clear color label → MaskEncoder → ref_embedding
修改後：clear RGB        → MaskEncoder → ref_embedding
```

MaskEncoder 接受 3-channel 輸入，介面完全相容，不需修改 MaskEncoder 本身。

**需要處理的問題：ref_void_mask 邏輯重設計**

現況邏輯依賴 color label 的特性：`ignore_index=255` 對應純黑（RGB 0,0,0），故 `sum(dim=0) == 0` 可識別 void。換成晴天 RGB 後此規律失效。

建議選項（擇一）：
- **選項 A（推薦）**：廢除 `ref_void_mask`，傳全 False tensor，CrossViewAlignment 看見完整晴天 RGB 所有 patch，最乾淨。
- **選項 B**：改用對應的晴天 GT labelTrainIds，將 class=255 的位置作為 void mask。

**CSV 修改**：`ref_mask_path` 欄位改指向晴天 RGB 路徑（需確認 ACDC 對應目錄結構，通常為 `rgb_ref/` 目錄下的 `_rgb_ref_anon.png`）。

---

## 5. 核心研究方向：Location-Aware Contrastive Loss

### 5.1 問題陳述

現有架構假設「晴天 ref 有幫助」，但沒有任何機制強制模型學習跨天氣的地點對應關係。模型完全有可能忽略 ref，只靠 `image_embedding` 本身做分割。要在論文中宣稱「模型學習到了跨天氣的地點不變表示」，需要一個顯式的學習目標。

### 5.2 核心概念

ACDC 的同地點 adverse/clear 配對天然構成對比學習的正負樣本：

```
正樣本對：(adverse_i, clear_i)  — 同地點，不同天氣
負樣本對：(adverse_i, clear_j)  — 不同地點，i ≠ j
```

Batch 內的所有樣本本身就構成完整的訓練訊號，不需要額外資料或特殊 batch 構建。

### 5.3 數學定義（InfoNCE Loss）

```
對 batch 中第 i 個樣本：

  z_adv_i  = proj( GAP(image_embedding_i) )   # (D,)，D=128
  z_clr_i  = proj( GAP(ref_embedding_i) )     # (D,)

  L_i = -log[ exp(sim(z_adv_i, z_clr_i) / τ)
              / Σ_j exp(sim(z_adv_i, z_clr_j) / τ) ]

  L_contrastive = (1/B) Σ_i L_i
```

- `GAP`：Global Average Pooling，(B, 256, 64, 64) → (B, 256)
- `proj`：2 層 MLP projection head，256 → 256 → 128
- `sim`：cosine similarity
- `τ`：temperature，初始值 0.07

### 5.4 實作位置

**`weather_sam.py` — 新增 projection head**
```python
# __init__ 中加入
self.contrastive_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
```

**`weather_sam.py` — forward 回傳對比特徵**
```python
# 在 ref_embeddings 計算後（約第 123 行）加入
contrast_feats = {
    "adverse": self.contrastive_proj(image_embeddings.mean(dim=[2, 3])),  # (B, 128)
    "clear":   self.contrastive_proj(ref_embeddings.mean(dim=[2, 3]))     # (B, 128)
}
# 加入 forward 回傳值
```

**`new_loss.py` — 新增 InfoNCE 函數**
```python
def location_contrastive_loss(adverse_feats, clear_feats, temperature=0.07):
    """
    adverse_feats: (B, D) — L2 正規化後的惡劣天氣 feature
    clear_feats:   (B, D) — L2 正規化後的晴天 feature
    對角線 = 正樣本，非對角線 = 負樣本
    """
    adverse_feats = F.normalize(adverse_feats, dim=-1)
    clear_feats   = F.normalize(clear_feats,   dim=-1)
    logits = torch.matmul(adverse_feats, clear_feats.T) / temperature  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
```

**`weather_trainer.py` — 加入 auxiliary loss**
```python
# Stage 5 之後，加入對比損失（auxiliary，不影響主路徑梯度）
contrast_loss = location_contrastive_loss(
    outputs_contrast["adverse"],
    outputs_contrast["clear"]
)
sample_total_loss += self.contrast_weight * contrast_loss  # 建議權重 0.1~0.2
```

### 5.5 此 Loss 帶來的研究貢獻

| 面向 | 加入前 | 加入後 |
|---|---|---|
| 模型是否學習地點對應 | 不確定（隱式） | 是（顯式學習目標） |
| 論文宣稱 | 「ref 有幫助」 | 「模型學到跨天氣的地點不變表示」 |
| 定量證據 | 只有 mIoU 提升 | 另可報告同地點 vs 不同地點的 cosine similarity 差距 |
| 修改主架構 | — | 否（純 auxiliary loss） |

**Batch size 建議**：batch size 越大，負樣本越多，效果越好。若受 GPU 記憶體限制，建議搭配 gradient accumulation，或參考 MoCo 的 momentum queue 維護更多負樣本。

---

## 6. 可解釋性驗證計畫

要在論文中證明「晴天 ref 幫助了惡劣天氣分割」，需要超越單純的 mIoU 數字，提供視覺化與定量的機制解釋。

### 6.1 GatedFusion Alpha 視覺化（最低成本，現在即可做）

GatedFusion 輸出的 alpha gate 是 `(1, H, W)` 的空間權重圖，直接代表模型在每個位置對 ref 的依賴程度。

**預期結果**：fog/rain/snow 場景中，能見度低的區域（遠景、霧化區）alpha 值應顯著高於能見度好的區域（近景清晰物件）。

**意義**：可以說明「模型在看不清楚的地方更依賴晴天先驗」，這是直接的人類可理解解釋。

### 6.2 CrossViewAlignment Attention Map 視覺化

CrossViewAlignment 的 cross-attention 權重（shape: B × heads × N_curr × N_ref）隱式包含了對應關係：adverse image 的每個 patch 對 clear ref 的哪些 patch 有高 attention。

**預期結果**：若地點對應有效，adverse 影像中道路區域的 attention 應集中在 clear ref 的道路區域，而非隨機分布。

**意義**：可視覺化「跨天氣的場景結構對應」，是最直接的對應關係證明。

### 6.3 Feature Space 相似度分析

**方法**：
1. 對同地點的 adverse/clear pair，計算 `cosine_similarity(image_embedding_i, ref_embedding_i)`
2. 對不同地點的 pair，計算 `cosine_similarity(image_embedding_i, ref_embedding_j)`
3. 比較兩個分布的差距（t-test 或分布圖）

**加入 Contrastive Loss 前後各做一次**，若訓練後同地點相似度顯著高於不同地點，即為地點對應的定量證明。

### 6.4 按類別與天氣類型的分層評估

| 分層 | 預期 | 用途 |
|---|---|---|
| 靜態類別（road, building, sky） | 地點先驗效果強 | 驗證靜態場景結構對應 |
| 動態類別（person, car, rider） | 效果弱（跨時間位置不對應） | 說明模型局限性，誠實呈現 |
| Fog | 地點先驗效果最強（能見度最低） | 找最佳應用場景 |
| Rain / Snow | 效果次之 | 天氣差異影響程度分析 |

---

## 7. 實驗設計與消融研究

### 7.1 Ablation 實驗矩陣

| 實驗編號 | Ref 輸入 | LocationEncoder | Contrastive Loss | 預期觀察 |
|---|---|---|---|---|
| A（現況 Cityscapes baseline） | color label | ✅ | ❌ | 基線 mIoU |
| B | color label | ❌ | ❌ | 驗證 LocationEncoder 貢獻 |
| C | 晴天 RGB | ❌ | ❌ | 驗證 RGB ref 的貢獻 |
| D | 晴天 RGB | ❌ | ✅ | 完整新架構 |
| E（對照組） | 打亂配對的晴天 RGB | ❌ | ❌ | 驗證地點對應是否真正有效 |

**實驗 E 是關鍵**：若 E 與 C 的 mIoU 接近，代表模型沒有利用地點對應，CrossViewAlignment 只是通用語義對齊。若 C 顯著優於 E，才能宣稱地點資訊有效。

### 7.2 ref 品質敏感度測試

刻意對 ref 影像施加退化（加高斯噪聲、降低解析度、部分遮擋），觀察 mIoU 下降幅度。品質敏感度越低，代表模型對 ref 的依賴越健康（輔助而非依賴）。

### 7.3 評估指標

- **主要指標**：mIoU（19 類別平均）
- **分層指標**：per-class IoU、per-weather-type mIoU
- **對應關係指標**：同地點 vs 不同地點 feature cosine similarity（加入 Contrastive Loss 前後對比）
- **Gate 分析**：alpha gate 在不同天氣強度下的平均值統計

---

## 8. 相關文獻方向

### 8.1 Retrieval-Augmented Segmentation

| 論文 | 關鍵詞 | 相關性 |
|---|---|---|
| Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching（2023） | retrieval-based segmentation, feature matching | 直接相關 |
| Video Object Segmentation using Space-Time Memory Networks（ICCV 2019, STM） | memory-based matching | 架構概念可借鑑 |
| Associating Objects with Transformers for Video Object Segmentation（NeurIPS 2021, AOT） | memory bank, cross-frame association | 記憶庫設計參考 |

**建議搜尋關鍵字**：
```
"retrieval augmented semantic segmentation"
"reference guided segmentation adverse weather"
"cross-condition feature alignment segmentation"
"memory-based segmentation weather"
```

### 8.2 天氣不變特徵學習

| 論文 | 關鍵詞 |
|---|---|
| FIFO: Learning Fog-invariant Features for Foggy Scene Understanding（CVPR 2022） | fog-invariant feature, domain generalization |
| RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening（CVPR 2021） | domain generalization, whitening |
| DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation（CVPR 2022） | domain adaptation, transformer |

### 8.3 Contrastive Learning for Dense Prediction

| 論文 | 關鍵詞 |
|---|---|
| DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training（CVPR 2021） | dense contrastive, pixel-level |
| PixPro: Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning（CVPR 2021） | pixel propagation, contrastive |

---

## 9. 執行優先順序

```
階段 1（現在可做，不需修改架構）
  ├── GatedFusion alpha 視覺化
  │     → 確認模型是否在惡劣區域依賴 ref
  └── CrossViewAlignment attention map 視覺化
        → 確認是否有地點對應的 attention 分布

階段 2（架構修改，低風險）
  ├── 移除 LocationEncoder
  │     → 清理無效模組，減少訓練雜訊
  └── ref 換成晴天 RGB
        → 需重設 ref_void_mask 邏輯，更新 CSV

階段 3（新功能，核心研究貢獻）
  └── 加入 Location-Aware Contrastive Loss
        → 將地點對應從隱式變顯式
        → 提供論文中的機制解釋證據

階段 4（評估與分析）
  ├── 執行 Ablation 實驗矩陣（A→B→C→D→E）
  ├── 分層評估（per-class / per-weather）
  └── Feature Space 相似度分析（定量地點對應證明）
```

**關鍵決策點**：階段 1 的視覺化結果決定階段 3 是否必要。若 attention map 已顯示良好的地點對應，Contrastive Loss 是錦上添花；若 attention map 混亂，Contrastive Loss 是修正模型學習方向的必要手段。

---

*本報告涵蓋截至 2026-04-11 的所有討論內容，包含已完成修改、待執行重構、核心研究方向與實驗計畫。*
