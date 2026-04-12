# WeatherSAM 研究報告書
**版本**：v2.0（Mask2Former-style）　｜　**日期**：2026-04-12　｜　**作者**：WeatherSAM Research

---

## 目錄

1. [專案概述](#1-專案概述)
2. [現有架構說明（Mask2Former v1）](#2-現有架構說明mask2former-v1)
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
| Cityscapes | 現有訓練基線 | `train_with_gps.csv`，使用 GPS LocationEncoder |
| ACDC | 惡劣天氣訓練目標 | fog / rain / snow，使用 ConditionEncoder（無 GPS） |

### 1.3 ACDC CSV 格式（`Datasets/acdc_train.csv`）

| 欄位 | 內容 | 用途 |
|---|---|---|
| `image_path` | 惡劣天氣 RGB（`_rgb_anon.png`） | ViT-H 輸入 |
| `ref_mask_path` | 晴天 color label（`_gt_ref_labelColor.png`） | MaskEncoder 輸入（現況）→ 未來改為晴天 RGB |
| `gt_path` | 惡劣天氣 labelTrainIds（`_gt_labelTrainIds.png`） | CrossEntropyLoss target |
| `lat` / `lon` | 填 0.0（ACDC 無 GPS） | ConditionEncoder 模式下不使用 |
| `condition_id` | 0=fog, 1=rain, 2=snow | ConditionEncoder 的輸入（**v2 新增**） |
| `invalid_mask` | ACDC 官方無效區域遮罩 | 移除固定盲區（車頭、相機遮擋） |

驗證結果：750 筆資料全數通過格式驗證（2026-04-11）。

---

## 2. 現有架構說明（Mask2Former v1）

> **本節描述的是 2026-04-12 版本的架構，已整合 Mask2Former-style 解碼機制。**
> 舊架構（v15/v16，per-class 獨立解碼 + 3-candidate + IoU 選擇）已保留於 git 歷史（commit `3a021e6`），可隨時回滾。

### 2.1 資料流

```
[輸入]
  adverse RGB (1024×1024)
  clear color label (ref_mask_path)
  text prompts (class names, K 個)
  GPS coords (lat, lon) 或 condition_id (fog=0 / rain=1 / snow=2)

[前向傳播]

  ── Stage 1: Multi-Modal Feature Extraction ──

  adverse RGB
    → preprocess (normalize + pad)
    → ViT-H ImageEncoder (Frozen)
    → image_embedding (B, 256, 64, 64)

  clear color label
    → MaskEncoder (Trainable, 4-layer CNN)
    → ref_embedding (B, 256, 64, 64)

  ── Stage 2: Cross-View Fusion ──

  image_embedding + ref_embedding
    → CrossViewAlignment (8-head cross-attention)
    → aligned_embedding (B, 256, 64, 64)

  image_embedding + aligned_embedding
    → GatedFusion (gate net + LayerNorm, 逐像素 α ∈ (0,1))
    → fused_embedding (B, 256, 64, 64)

  ── Stage 3: Prompt Encoding ──

  text prompts (K 個類別名稱)
    → TextEncoder (CLIP ViT-B/32, Frozen + Trainable projection)
    → sparse_embeddings (K, 1, 256)

  GPS coords (Cityscapes) 或 condition_id (ACDC)
    → LocationEncoder / ConditionEncoder
    → location_embeddings (K, 1, 256)

  sparse_embeddings + location_embeddings
    → WeatherPromptEncoder (concat → K, 2, 256)

  ── Stage 4: Mask2Former-Style Decoding (核心改動) ──

  class_mask_tokens: 19 個類別各自獨立的 learnable query (K, 256)
  tokens = [class_q₀, ..., class_qₖ, prompt₀, ..., promptₖ]  (1, K + K×2, 256)

  fused_embedding + prompt_embeddings + class_mask_tokens
    → TwoWayTransformer (Single Forward Pass)
    → mask_token_out (1, K, 256)

  mask_token_out
    → 19 個專屬 class_hypernetworks_mlps
    → low_res_logits (1, K, 256, 256)  ← 每類別 1 張 mask

  ── Stage 5: Residual Pixel Refinement ──

  low_res_logits (組裝 19-channel，缺席類別填 -10.0)
    → ResidualDWConvFusion (3×3 DW Conv + 1×1 PW Conv + residual)
    → fused_logits (1, 19, 256, 256)
    → postprocess_masks → fused_logits_hr (1, 19, 1024, 1024)

  ── Stage 6: Loss Computation ──

  MaskLoss    = Focal Loss + Dice Loss  （per-class binary，直接使用 1 mask）
  ContextLoss = CrossEntropyLoss (class-balanced, ignore_index=255)
  ABL         = Active Boundary Loss（Epoch ≥ 35 後啟動）
```

### 2.2 與 v15/v16 架構的關鍵差異

| 項目 | v15/v16（舊架構） | Mask2Former v1（現架構） |
|------|-------------------|------------------------|
| Transformer 呼叫 | 每類別各 1 次（共 K 次） | 所有 K 類別**單次 forward** |
| Query Token | 4 個共享 `mask_tokens` + 1 個 `iou_token` | 19 個獨立 `class_mask_tokens` |
| 類別間互動 | 無 | 有（cross-class self-attention） |
| Mask 候選數 | 每類別 3 個，IoU head 選最佳 | 每類別直接 1 個 |
| Hypernetwork | 4 個共享 MLP | 19 個專屬 MLP |
| IoU MSE Loss | 需要 | **移除** |
| Fusion Head | ContextFusionHead（~數萬 params） | ResidualDWConvFusion（~589 params） |
| 天氣條件編碼 | 僅 GPS LocationEncoder | 新增 ConditionEncoder（ACDC 模式） |

### 2.3 Ignore Index 處理機制

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

### 2.4 訓練策略

| 機制 | 設定 | 說明 |
|------|------|------|
| LR Schedule | Warmup (5 epoch) + Cosine Decay | LambdaLR |
| Gradient Detach | epoch 0–4 | ResidualDWConvFusion 接收 detached logits，防止初期梯度爆炸 |
| Adam Momentum Reset | epoch 5 | 解開 detach 時重置 class_mask_tokens / class_hypernetworks_mlps / output_upscaling 的 Adam state |
| ABL 延遲啟動 | epoch ≥ 35 | 讓 mask 先穩定學習，ABL 啟動時重置 early stop 計數器 |
| Gradient Accumulation | ×4 | 等效 batch_size = 8 |
| AMP | torch.amp.autocast | 混合精度訓練 |
| Gradient Clipping | max_norm = 1.0 | 防止梯度爆炸 |
| Early Stopping | patience = 10, min_delta = 0.005 | ABL 啟動時自動重置 |

### 2.5 損失函數

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{ctx}} + \lambda_{\text{abl}} \mathcal{L}_{\text{abl}} \cdot \mathbb{1}[t \geq 35]$$

| Loss | 權重 | 說明 |
|---|---|---|
| MaskLoss (Focal) | 4.0 | Per-class binary mask，v15/v16 為 20.0（降低因新架構梯度更穩定） |
| MaskLoss (Dice) | 1.5 | Per-class binary mask |
| ~~IoU MSE~~ | ~~移除~~ | ~~v15/v16 需要（3-candidate 選擇）；v2 每類別 1 mask，無下游用途~~ |
| ContextLoss (CE) | 1.0 | 全域 19-class cross-entropy，class-balanced |
| ABL | 0.5, epoch ≥ 35 | 6-phase boundary alignment loss |

### 2.6 可訓練模組

| Module | LR Scale | Role |
|---|---|---|
| `fusion_module` | 1× | CrossViewAlignment |
| `gate_module` | 1× | GatedFusion |
| `location_encoder.output_projection` / `condition_encoder` | 1× | GPS / 天氣條件編碼 |
| `text_encoder.projection` | 1× | CLIP text → SAM space |
| `context_fusion_head` (ResidualDWConvFusion) | 1× | Pixel-level 殘差精修 |
| `mask_encoder` | 1× | 參考遮罩編碼 |
| `mask_decoder.class_mask_tokens` | 1× | 19 per-class query embeddings |
| `mask_decoder.class_hypernetworks_mlps` | 1× | 19 per-class mask weight generators |
| `mask_decoder.output_upscaling` | 1× | 64×64 → 256×256 feature upsampling |
| `mask_decoder.iou_token` / `mask_tokens` | 0.1× | SAM 原始 tokens（保留向後兼容） |
| `mask_decoder.transformer` | 0.01× | TwoWayTransformer（極低 LR 適應 fused feature） |
| `pe_layer` | 1× | 位置編碼 |

**完全凍結**：`image_encoder` (ViT-H)、`text_encoder` (CLIP backbone)、`location_encoder` (GeoCLIP backbone)

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

### 3.2 資料驗證工具（2026-04-11）

- **`validate_acdc_csv.py`**：驗證 CSV 中每筆資料的三個欄位格式正確性
  - `image_path`：3-channel RGB 可讀
  - `ref_mask_path`：3-channel color PNG，非全黑
  - `gt_path`：pixel 值 ∈ {0..18, 255}
  - 結果：750/750 通過

- **`visualize_acdc_sample.py`**：隨機抽樣視覺化，三欄並排（adverse RGB / clear color label / GT colorized）

### 3.3 Mask2Former-Style 解碼架構（2026-04-12）⭐

**核心改動**：將 SAM 的 per-class 獨立解碼替換為 Mask2Former 風格的統一查詢解碼。

**學術依據**：Mask2Former (Cheng et al., CVPR 2022) — per-object learnable query + unified transformer decoding。

| 檔案 | 修改內容 |
|---|---|
| `weather_mask_decoder.py` | 新增 `class_mask_tokens` (nn.Embedding(19, 256))、`class_hypernetworks_mlps` (19 個獨立 MLP)、`forward_semantic()` / `predict_masks_semantic()` 方法 |
| `weather_sam.py` | forward 改呼叫 `forward_semantic()`，output dict 移除 `iou_predictions`，新增 `CLASS_MAP` (19 class) |
| `build_weather_sam.py` | 新增 `num_classes=19` 傳入 MaskDecoder |

**設計特點**：
- 所有 K 個 active class query 在同一 TwoWayTransformer sequence 中處理
- Cross-class self-attention 使類別間具備互斥感知能力
- 每類別直接輸出 1 張 mask，無需 3-candidate IoU 選擇
- `dense_prompt_embeddings[:1]` 修正 batch dim 不匹配問題

### 3.4 IoU MSE Loss 完整移除（2026-04-12）

**移除原因**：v15/v16 中 IoU head 用於 3 個候選 mask 的評分選擇。新架構每類別直接輸出 1 張 mask，IoU head 無任何下游用途。

| 檔案 | 移除內容 |
|---|---|
| `weather_mask_decoder.py` | `class_iou_tokens` 從 token 序列移除、iou prediction 計算移除、回傳型別改為 `torch.Tensor` |
| `weather_sam.py` | output dict 移除 `"iou_predictions"` key |
| `weather_trainer.py` | Stage 2 IoU MSE 區塊、`iou_mse_loss_fn`、`iou_weight`、`calculate_true_iou` import（全部以註解保留） |
| `weather_predictor.py` | 回傳簽名從 `(masks, iou_predictions, low_res_masks, class_ids)` → `(masks, low_res_masks, class_ids)` |
| `train.py` | `--iou_weight` 參數、log_entry、print、plot_history（全部以註解保留） |

### 3.5 ResidualDWConvFusion 取代 ContextFusionHead（2026-04-12）

**設計理由**：Mask2Former self-attention 已在 token 空間處理類別競爭，ContextFusionHead 的 Spatial-Aware Channel Attention 功能與 CE loss 的 softmax 重疊。ResidualDWConvFusion 只補足 **pixel 空間的局部空間一致性**。

| 項目 | ContextFusionHead（舊） | ResidualDWConvFusion（新） |
|------|------------------------|---------------------------|
| 結構 | InstanceNorm → 多層 Conv → 4×4 Spatial Attention → Classifier → residual_scale(0.1) | 3×3 DW Conv → 1×1 PW Conv (zero-init) → GroupNorm → residual |
| 參數量 | ~數萬 | ~589 |
| 初始行為 | 接近 identity (0.1 scale) | 完全 identity (zero-init) |
| 與 CE loss 重疊 | 高 | 低 |

| 檔案 | 修改內容 |
|---|---|
| `fusion_head.py` | 新增 `ResidualDWConvFusion`，舊 `ContextFusionHead` 完整保留 |
| `weather_sam.py` | import 與實例化替換（舊版以註解保留） |

### 3.6 ACDC ConditionEncoder 支援（2026-04-12）

**動機**：ACDC 的 GPS 座標全為 (0, 0)，LocationEncoder 接收無意義的輸入。以天氣條件 Embedding 取代。

| 檔案 | 修改內容 |
|---|---|
| `weather_sam.py` | 新增 `self.condition_encoder = nn.Embedding(3, 256)`，forward 中 `condition_id >= 0` 啟用 ConditionEncoder |
| `weather_dataloader.py` | 新增 `condition_id` 欄位讀取，collate_fn 加入 `condition_ids` |
| `weather_trainer.py` | `use_condition_embedding` flag 控制 param group（ConditionEncoder vs LocationEncoder.output_projection） |
| `train.py` | 新增 `--use_condition_embedding` argparse（store_true, default=False） |

**模式切換**：
```
--use_condition_embedding=False (預設)  → Cityscapes 模式，使用 LocationEncoder
--use_condition_embedding=True          → ACDC 模式，使用 ConditionEncoder
```

### 3.7 訓練 Param Group 更新（2026-04-12）

**main_lr_modules 變更**：

| 移除 | 新增 |
|------|------|
| `mask_decoder.iou_prediction_head` | `mask_decoder.class_mask_tokens` |
| `mask_decoder.output_hypernetworks_mlps` | `mask_decoder.class_hypernetworks_mlps` |

**Adam Momentum Reset（epoch 5）更新**：

| 移除 | 新增 |
|------|------|
| `iou_prediction_head` | `class_mask_tokens` |
| `output_hypernetworks_mlps` | `class_hypernetworks_mlps` |
| `iou_token`, `mask_tokens` | — |
| — | `output_upscaling`（保留） |

### 3.8 Loss 權重調整（2026-04-12）

| Loss | v15/v16 | Mask2Former v1 | 調整原因 |
|------|---------|---------------|---------|
| Focal | 20.0 | 4.0 | 新架構無 3-candidate 競爭，梯度更穩定 |
| Dice | 1.0 | 1.5 | 微調 |
| IoU MSE | 1.0 | **移除** | 無候選選擇需求 |
| ABL start | epoch 5 | epoch 35 | 100 epoch 訓練，35% 開始 |

---

## 4. 待執行的架構重構

### 4.1 移除 LocationEncoder（部分完成）

**現況**：已新增 ConditionEncoder 作為 ACDC 模式的替代。LocationEncoder 在 Cityscapes 模式下仍然啟用。

**完整移除的動機**：LocationEncoder 以 GPS 座標作為地點表示，在 ACDC 750 筆資料量下，座標→場景的映射無法有效學習。地點資訊應以視覺形式（晴天 RGB）直接輸入，而非符號形式（座標）。

**剩餘工作**：若 ACDC 實驗確認 ConditionEncoder 有效，可考慮完全封存 LocationEncoder。

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

### 4.3 ResidualDWConvFusion 加入 Image Feature Cross-Attention（未來方向）

**動機**：目前的 ResidualDWConvFusion 只看 logit map，不看 image feature。若加入 cross-attention 回 image feature，可以提供 Mask2Former self-attention 完全沒有的空間細節資訊。

**設計方向**：
```
logit map (1, 19, 256, 256) + image_feature (1, 256, 64, 64)
  → cross-attention: logit map attend to image feature
  → 精修後的 logit map
```

**時機**：在 ref 改為晴天 RGB 後，此模組也是接收 ref image feature 的自然位置。

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
# 在 ref_embeddings 計算後加入
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

### 6.3 Mask2Former Cross-Class Self-Attention 分析（新增）

新架構中，class query tokens 之間的 self-attention 權重可以揭示類別間的競爭關係。

**預期結果**：road 的 query 對 car 的 query 應有較高 attention（因為它們經常相鄰），而對 sky 的 query 較低。

**意義**：可驗證 Mask2Former-style 解碼是否確實學到了有意義的類別互斥關係。

### 6.4 Feature Space 相似度分析

**方法**：
1. 對同地點的 adverse/clear pair，計算 `cosine_similarity(image_embedding_i, ref_embedding_i)`
2. 對不同地點的 pair，計算 `cosine_similarity(image_embedding_i, ref_embedding_j)`
3. 比較兩個分布的差距（t-test 或分布圖）

**加入 Contrastive Loss 前後各做一次**，若訓練後同地點相似度顯著高於不同地點，即為地點對應的定量證明。

### 6.5 按類別與天氣類型的分層評估

| 分層 | 預期 | 用途 |
|---|---|---|
| 靜態類別（road, building, sky） | 地點先驗效果強 | 驗證靜態場景結構對應 |
| 動態類別（person, car, rider） | 效果弱（跨時間位置不對應） | 說明模型局限性，誠實呈現 |
| Fog | 地點先驗效果最強（能見度最低） | 找最佳應用場景 |
| Rain / Snow | 效果次之 | 天氣差異影響程度分析 |

---

## 7. 實驗設計與消融研究

### 7.1 Ablation 實驗矩陣

| 實驗編號 | 解碼架構 | Ref 輸入 | LocationEncoder | Contrastive Loss | 預期觀察 |
|---|---|---|---|---|---|
| A（v15 Cityscapes baseline） | per-class 獨立 | color label | ✅ | ❌ | 基線 mIoU |
| B（Mask2Former baseline） | Mask2Former-style | color label | ✅ | ❌ | 驗證 Mask2Former 架構貢獻 |
| C | Mask2Former-style | color label | ❌ | ❌ | 驗證 LocationEncoder 貢獻 |
| D | Mask2Former-style | 晴天 RGB | ❌ | ❌ | 驗證 RGB ref 的貢獻 |
| E | Mask2Former-style | 晴天 RGB | ❌ | ✅ | 完整新架構 |
| F（對照組） | Mask2Former-style | 打亂配對的晴天 RGB | ❌ | ❌ | 驗證地點對應是否真正有效 |

**實驗 F 是關鍵**：若 F 與 D 的 mIoU 接近，代表模型沒有利用地點對應，CrossViewAlignment 只是通用語義對齊。若 D 顯著優於 F，才能宣稱地點資訊有效。

**實驗 A vs B 是 Mask2Former 架構的直接貢獻驗證**。

### 7.2 ref 品質敏感度測試

刻意對 ref 影像施加退化（加高斯噪聲、降低解析度、部分遮擋），觀察 mIoU 下降幅度。品質敏感度越低，代表模型對 ref 的依賴越健康（輔助而非依賴）。

### 7.3 評估指標

- **主要指標**：mIoU（19 類別平均）
- **分層指標**：per-class IoU、per-weather-type mIoU
- **對應關係指標**：同地點 vs 不同地點 feature cosine similarity（加入 Contrastive Loss 前後對比）
- **Gate 分析**：alpha gate 在不同天氣強度下的平均值統計
- **Cross-class attention**：Mask2Former query 間的 self-attention 權重分析

---

## 8. 相關文獻方向

### 8.1 Mask2Former 與統一分割架構

| 論文 | 關鍵詞 | 相關性 |
|---|---|---|
| Mask2Former (Cheng et al., CVPR 2022) | per-object query, unified segmentation | **核心依據** — class_mask_tokens + unified transformer decoding |
| MaskFormer (Cheng et al., NeurIPS 2021) | per-pixel classification → mask classification | Mask2Former 的前身 |
| OneFormer (Jain et al., CVPR 2023) | task-conditioned joint training | 統一分割的進一步發展 |

### 8.2 Retrieval-Augmented Segmentation

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

### 8.3 天氣不變特徵學習

| 論文 | 關鍵詞 |
|---|---|
| FIFO: Learning Fog-invariant Features for Foggy Scene Understanding（CVPR 2022） | fog-invariant feature, domain generalization |
| RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening（CVPR 2021） | domain generalization, whitening |
| DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation（CVPR 2022） | domain adaptation, transformer |

### 8.4 Contrastive Learning for Dense Prediction

| 論文 | 關鍵詞 |
|---|---|
| DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training（CVPR 2021） | dense contrastive, pixel-level |
| PixPro: Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning（CVPR 2021） | pixel propagation, contrastive |

---

## 9. 執行優先順序

```
階段 0（進行中）
  └── Cityscapes Mask2Former v1 訓練驗證
        → 確認新架構（class_mask_tokens + ResidualDWConvFusion）
          能否正常收斂並達到合理 mIoU
        → 與 v15/v16 基線比較（git checkout 3a021e6 可回滾）

階段 1（架構驗證後可做，不需修改架構）
  ├── GatedFusion alpha 視覺化
  │     → 確認模型是否在惡劣區域依賴 ref
  ├── CrossViewAlignment attention map 視覺化
  │     → 確認是否有地點對應的 attention 分布
  └── Mask2Former cross-class self-attention 分析
        → 確認類別間互斥關係是否合理

階段 2（ACDC fine-tune）
  └── 使用 Cityscapes Mask2Former v1 checkpoint
      → ACDC fine-tune（--use_condition_embedding）
      → 驗證 ConditionEncoder 有效性

階段 3（架構修改，低風險）
  ├── 移除 LocationEncoder（完整封存）
  │     → 清理無效模組，減少訓練雜訊
  └── ref 換成晴天 RGB
        → 需重設 ref_void_mask 邏輯，更新 CSV

階段 4（新功能，核心研究貢獻）
  └── 加入 Location-Aware Contrastive Loss
        → 將地點對應從隱式變顯式
        → 提供論文中的機制解釋證據

階段 5（評估與分析）
  ├── 執行 Ablation 實驗矩陣（A→B→C→D→E→F）
  ├── 分層評估（per-class / per-weather）
  └── Feature Space 相似度分析（定量地點對應證明）
```

**關鍵決策點**：
- **階段 0 結果**決定是否保留 Mask2Former 架構或回滾 v15/v16。
- **階段 1 的視覺化結果**決定階段 4 是否必要。若 attention map 已顯示良好的地點對應，Contrastive Loss 是錦上添花；若 attention map 混亂，Contrastive Loss 是修正模型學習方向的必要手段。

---

## 附錄：版本回滾指南

```bash
# 回滾至 v15/v16 舊架構
git checkout 3a021e6 -- .

# 回到 Mask2Former v1 架構
git checkout main -- .
```

---

*本報告涵蓋截至 2026-04-12 的所有討論內容，包含 Mask2Former-style 架構升級、IoU Loss 移除、ResidualDWConvFusion、ConditionEncoder、待執行重構、核心研究方向與實驗計畫。*
