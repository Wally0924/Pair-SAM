# WeatherSAM v2: Mask2Former-Style Semantic Decoding with SAM Adaptation

<div align="center">

> **WeatherSAM v2：基於 Mask2Former 類別查詢機制與 SAM 適應性架構的惡劣天氣語意分割框架**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Changelog / 版本更新紀錄

> **v2（Mask2Former-style 架構升級）** 相較於 v1 的核心變更：

| 項目 | v1（舊架構） | v2（新架構） |
|------|-------------|-------------|
| Mask 解碼方式 | 每類別各跑一次 Transformer，產生 3 個候選 mask，IoU head 選最佳 | 所有 K 個類別在**單次 Transformer forward** 中處理，每類別直接輸出 1 張 mask |
| Query Token | 4 個共享 `mask_tokens`（所有類別用同一組） | 19 個獨立 `class_mask_tokens`（每類別專屬 learnable query） |
| 類別間互動 | 無（完全獨立解碼） | 有（cross-class self-attention，類別間互斥感知） |
| IoU Loss | 需要（評分選最佳候選） | **移除**（無候選選擇需求） |
| Fusion Head | ContextFusionHead（InstanceNorm + 多層 Conv + Spatial Attention） | **ResidualDWConvFusion**（輕量 DW+PW Conv + zero-init 殘差） |
| 天氣條件編碼 | 僅 GPS LocationEncoder | 新增 **ConditionEncoder**（fog/rain/snow embedding），支援 ACDC |
| Hypernetwork | 4 個共享 MLP | 19 個類別各自的專屬 MLP |

---

## Abstract / 摘要

**[EN]**
We present **WeatherSAM v2**, an evolution of the WeatherSAM framework that replaces the per-class independent decoding paradigm with a **Mask2Former-style unified query mechanism**. By introducing 19 class-specific learnable query tokens that participate in a **single shared Transformer forward pass**, cross-class self-attention naturally enforces mutual exclusivity — eliminating the need for IoU-based candidate selection and reducing the Context Fusion Head to a lightweight residual refinement module. The framework additionally introduces a **ConditionEncoder** for weather-condition-aware domain adaptation on the ACDC benchmark (fog, rain, snow). Built upon SAM's frozen ViT-H backbone, WeatherSAM v2 achieves improved architectural elegance while maintaining parameter efficiency (~10–20% trainable parameters).

**[ZH]**
本文提出 **WeatherSAM v2**，將原本各類別獨立解碼的範式升級為 **Mask2Former 風格的統一查詢機制**。透過引入 19 個類別專屬的可學習查詢 Token，讓所有類別在**單次共享 Transformer forward pass** 中處理，跨類別的 self-attention 自然實現互斥感知——無需 IoU 候選選擇機制，且將 Context Fusion Head 簡化為輕量殘差精修模組。此外，新增 **ConditionEncoder** 以支援 ACDC 基準資料集（霧、雨、雪）的天氣條件感知領域自適應。基於 SAM 凍結的 ViT-H 骨幹，WeatherSAM v2 在保持參數高效性（約 10–20% 可訓練參數）的同時，大幅提升架構的簡潔性與學術嚴謹度。

---

## Table of Contents / 目錄

- [Architecture Overview](#architecture-overview--架構概觀)
- [Key Differences from v1](#key-differences-from-v1--與-v1-的關鍵差異)
- [Novel Contributions](#novel-contributions--創新貢獻)
- [Installation](#installation--環境安裝)
- [Dataset Preparation](#dataset-preparation--資料集準備)
- [Training](#training--訓練)
- [Inference](#inference--推論)
- [Results](#results--實驗結果)
- [Citation](#citation--引用)

---

## Architecture Overview / 架構概觀

WeatherSAM v2 採用 **六階段管線**，核心改動集中在第四與第五階段：

```
 Foggy Image ─────────► [Image Encoder (ViT-H, Frozen)] ──────────────────────────┐
 Reference Mask ──────► [Mask Encoder (Trainable)] ──────► [CrossViewAlignment] ──► [GatedFusion]
 GPS / Condition ─────► [LocationEncoder | ConditionEncoder] ────────────────────────────────► [WeatherPromptEncoder]
 Class Text Prompts ──► [Text Encoder (CLIP, Frozen Backbone)] ──────────────────────► [WeatherPromptEncoder]
                                                                                                          │
                                              ┌───────────────────────────────────────────────────────────┘
                                              ▼
                                    [Mask Decoder — Mask2Former-style]
                                    ─ 19 class-specific query tokens
                                    ─ Single Transformer forward (cross-class self-attention)
                                    ─ 19 per-class hypernetwork MLPs
                                    ─ 1 mask per class (K × 256×256)
                                              │
                                    [Postprocess: Bilinear ↑ to 1024×1024]
                                              │
                                    [ResidualDWConvFusion (Trainable)]
                                    ─ 3×3 DW Conv → 1×1 PW Conv → GroupNorm → Residual
                                              │
                                    [Argmax → Semantic Prediction (19 classes, 1024×1024)]
```

### Stage 1 — Multi-Modal Feature Extraction / 第一階段：多模態特徵萃取

| Component | Status | Input | Output | Details |
|---|---|---|---|---|
| Image Encoder (ViT-H) | Frozen | Foggy image 1024×1024 | 256×64×64 | SAM pre-trained backbone |
| Mask Encoder | Trainable | Clear-weather reference mask | 256×64×64 | Geometric prior encoding |
| Location Encoder | Frozen + Trainable proj. | GPS (lat, lon) | 256-dim | GeoCLIP + Equal-Earth (Cityscapes mode) |
| Condition Encoder | Trainable | condition_id (0/1/2) | 256-dim | `nn.Embedding(3, 256)` (ACDC mode) |
| Text Encoder | Frozen + Trainable proj. | Class name strings | 256-dim × K | CLIP ViT-B/32 |

### Stage 2 — Cross-View Fusion / 第二階段：跨視角融合

與 v1 相同。**CrossViewAlignment** 以 8 頭交叉注意力對齊晴天幾何先驗，**GatedFusion** 學習逐像素混合權重 α ∈ (0,1)。

$$\mathbf{f}_{\text{fused}} = (1-\alpha) \cdot \mathbf{f}_{\text{curr}} + \alpha \cdot \mathbf{f}_{\text{align}}$$

### Stage 3 — Prompt Encoding / 第三階段：提示編碼

每個類別的 sparse prompt = `(text_token, location_token)` → `(K, 2, 256)`。

- **Cityscapes 模式**：`location_token` 來自 LocationEncoder（GPS）
- **ACDC 模式**（`--use_condition_embedding`）：`location_token` 來自 ConditionEncoder（fog=0, rain=1, snow=2）

### Stage 4 — Mask2Former-Style Decoding / 第四階段：Mask2Former 風格解碼 ⭐

**此階段為 v2 的核心改動。**

```
                    ┌── class_mask_tokens ──────────────────┐
                    │   19 個類別各自獨立的 query embedding    │
                    │   class_q: (K, 256)                    │
                    └────────────────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────────────┐
                    │   Token 序列組裝：                       │
                    │   tokens = [class_q₀, ..., class_qₖ,   │
                    │             prompt₀, ..., promptₖ]      │
                    │   shape: (1, K + K×N_tok, 256)          │
                    └───────────────┬────────────────────────┘
                                    │
                    ┌───────────────▼────────────────────────┐
                    │   TwoWayTransformer (Single Forward)    │
                    │                                         │
                    │   • cross-attention: query ↔ image      │
                    │   • self-attention:  query ↔ query      │
                    │     ↑ 這裡是關鍵：car 的 query 看得到    │
                    │       road、building 的 query，形成     │
                    │       跨類別互斥感知                     │
                    └───────────────┬────────────────────────┘
                                    │
                    ┌───────────────▼────────────────────────┐
                    │   mask_token_out = hs[:, :K, :]         │
                    │   (1, K, 256)                           │
                    └───────────────┬────────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
       class_hyper[0]         class_hyper[k]         class_hyper[18]
        (MLP: 256→32)         (MLP: 256→32)         (MLP: 256→32)
             │                      │                      │
          w₀ @ upscaled         wₖ @ upscaled         w₁₈ @ upscaled
             │                      │                      │
        mask₀ (256×256)       maskₖ (256×256)       mask₁₈ (256×256)
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
                              masks: (1, K, 256, 256)
```

**學術依據**：Mask2Former (Cheng et al., CVPR 2022) 提出的 per-object learnable query + unified transformer decoding。

### Stage 5 — Residual Pixel Refinement / 第五階段：殘差像素精修

**v2 以 ResidualDWConvFusion 取代原本的 ContextFusionHead。**

```
19 張 logit map (1, 19, 256, 256)
         │
         ├──────────────────── identity (殘差分支)
         │
    3×3 DW Conv              每個 class channel 獨立空間平滑
         │                    （修補 mask 邊緣破碎處）
    1×1 PW Conv              跨類別線性混合
         │                    （zero-init → 訓練初期輸出為 0）
    GroupNorm
         │
     + identity              殘差相加 → 初始時等同 identity
         │
  缺席通道 → 恢復 -10.0
         │
精修後的 19 張 logit map
```

**設計理由**：
- Mask2Former 的 self-attention 已在 token 空間處理了類別競爭
- ResidualDWConvFusion 只補足 **pixel 空間的局部空間一致性**（self-attention 做不到的事）
- 參數量從 ContextFusionHead 的數萬降至 **~589 個**

### Stage 6 — Loss Computation / 第六階段：損失計算

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{ctx}} + \lambda_{\text{abl}} \mathcal{L}_{\text{abl}} \cdot \mathbb{1}[t \geq t_{\text{abl}}]$$

| Loss | Weight | Description |
|---|---|---|
| $\mathcal{L}_{\text{mask}}$ (Focal + Dice) | focal=4.0, dice=1.5 | Per-class binary mask quality |
| ~~$\mathcal{L}_{\text{iou}}$ (MSE)~~ | ~~removed~~ | ~~v1 IoU prediction head — v2 已移除~~ |
| $\mathcal{L}_{\text{ctx}}$ (Weighted CE) | 1.0 | Global 19-class semantic CE with class balancing |
| $\mathcal{L}_{\text{abl}}$ (Active Boundary) | 0.5, starts epoch 35 | Boundary alignment loss |

**IoU Loss 移除原因**：v1 中 IoU head 用於在 3 個候選 mask 中選最佳，v2 每類別直接輸出 1 張 mask，無候選選擇需求，IoU head 無任何下游用途。

---

## Key Differences from v1 / 與 v1 的關鍵差異

### 1. 類別查詢機制（核心改動）

```
v1：每個類別獨立解碼                    v2：所有類別統一解碼 (Mask2Former-style)
─────────────────────────               ─────────────────────────────────────────

  class_0 → Transformer → 3 masks      [class_0, class_1, ..., class_K]
  class_1 → Transformer → 3 masks         ↓ (同一個 Transformer)
  class_2 → Transformer → 3 masks      cross-class self-attention → 互斥感知
  ...                                      ↓
  class_K → Transformer → 3 masks      [mask_0, mask_1, ..., mask_K]  (各 1 張)
              ↓
  IoU head 選最佳 → K 張 mask           無需 IoU head，無需候選選擇

Transformer 呼叫次數：K 次              Transformer 呼叫次數：1 次
類別間互動：無                          類別間互動：有 (self-attention)
```

### 2. Fusion Head 簡化

```
v1 ContextFusionHead                    v2 ResidualDWConvFusion
─────────────────────                   ────────────────────────

InstanceNorm                            3×3 DW Conv (空間平滑)
  ↓                                       ↓
1×1 Conv → GN → ReLU                   1×1 PW Conv (跨類別混合, zero-init)
  ↓                                       ↓
3×3 DW Conv → GN → ReLU                GroupNorm
  ↓                                       ↓
1×1 Conv                               + identity
  ↓
4×4 Spatial Pool → Channel Attention
  ↓
Bilinear Upsample → feat × attn
  ↓
1×1 Classifier
  ↓
out × residual_scale(0.1) + identity

參數量：~數萬                           參數量：~589
```

### 3. ACDC 天氣條件支援（新增）

```
Cityscapes 模式 (--use_condition_embedding=False)
  GPS 座標 → LocationEncoder → 256-dim location token

ACDC 模式 (--use_condition_embedding=True)
  condition_id → ConditionEncoder (nn.Embedding(3, 256))
  fog=0, rain=1, snow=2 → 256-dim condition token
```

---

## Novel Contributions / 創新貢獻

### 1. Mask2Former-Style Unified Query Decoding / 統一查詢解碼

**[EN]** WeatherSAM v2 replaces per-class independent Transformer passes with a single unified forward pass where all K class-specific learnable query tokens attend to each other via self-attention. This cross-class interaction enables natural mutual exclusivity — a pixel is unlikely to be simultaneously predicted as "car" and "road" because their query tokens have competed for attention resources within the same Transformer layer. This eliminates the need for IoU-based candidate selection (3→1) and reduces Transformer invocations from K to 1.

**[ZH]** WeatherSAM v2 以單次統一 Transformer forward pass 取代各類別獨立的多次呼叫，讓所有 K 個類別的專屬查詢 Token 透過 self-attention 互相感知。這種跨類別互動自然實現互斥性——同一像素不太可能同時被預測為「car」與「road」，因為兩者的查詢 Token 已在 Transformer 內部競爭注意力資源。此設計消除了 IoU 候選選擇機制，並將 Transformer 呼叫從 K 次降為 1 次。

### 2. ResidualDWConvFusion / 殘差深度可分離精修

**[EN]** With cross-class competition handled in token space by self-attention, the pixel-level refinement head is simplified to a residual Depthwise Separable Convolution block. The 3×3 depthwise convolution smooths each class mask independently in spatial domain, while the 1×1 pointwise convolution learns inter-class suppression at each pixel. Zero-initialization of the pointwise weights ensures the module starts as identity, preventing gradient shocks and allowing stable convergence.

**[ZH]** 由於 self-attention 已在 Token 空間處理了跨類別競爭，Pixel-level 精修頭被簡化為殘差深度可分離卷積區塊。3×3 Depthwise Conv 對每個類別的 mask 在空間域獨立平滑，1×1 Pointwise Conv 學習各像素位置上的跨類別抑制。Pointwise 權重的 zero-init 確保模組初始行為等同 identity，避免梯度衝擊。

### 3. Cross-View Alignment + Gated Fusion / 跨視角對齊與閘控融合

（與 v1 相同）利用多頭交叉注意力對齊晴天參考特徵，以可學習閘控機制動態調整依賴比例。

### 4. Active Boundary Loss (ABL) / 主動邊界損失

（與 v1 相同）六階段可微邊界損失，以距離加權 KL 散度驅動預測邊界向真實邊界靠攏。

### 5. ConditionEncoder for ACDC / 天氣條件編碼器（新增）

**[EN]** For the ACDC benchmark where GPS coordinates are unavailable (all zeros), we introduce a lightweight ConditionEncoder — a 3-class learnable embedding (`nn.Embedding(3, 256)`) mapping fog/rain/snow conditions to 256-dim tokens. This replaces the LocationEncoder in the prompt pipeline via the `--use_condition_embedding` flag, enabling a two-stage training strategy: Cityscapes pre-training → ACDC fine-tuning.

**[ZH]** 針對 ACDC 基準資料集中 GPS 座標全為 0 的問題，引入輕量 ConditionEncoder——3 類可學習嵌入（`nn.Embedding(3, 256)`），將霧/雨/雪條件映射為 256 維 Token。透過 `--use_condition_embedding` 旗標在 prompt pipeline 中取代 LocationEncoder，支援兩階段訓練策略：Cityscapes 預訓練 → ACDC 微調。

### 6. Progressive Training Strategy / 漸進式訓練策略

| 機制 | Epoch | 說明 |
|------|-------|------|
| Gradient Detachment | 1–4 | ContextFusionHead 接收 detached logits，防止初期梯度爆炸 |
| Adam Momentum Reset | 5 | 解開 detach 時重置 Mask Decoder 模組的 Adam state，防止動量衝擊 |
| Warmup LR | 1–5 | 線性 LR warmup |
| Cosine Decay | 6–100 | 餘弦退火 |
| ABL Activation | 35+ | Active Boundary Loss 延遲啟動，讓 mask 先穩定學習 |

---

## Installation / 環境安裝

### Requirements / 系統需求

- Python ≥ 3.8
- PyTorch ≥ 2.0 with CUDA support
- NVIDIA GPU with ≥ 24 GB VRAM (recommended: A100 / RTX 3090 / RTX 4090)

### Setup / 安裝步驟

```bash
# Clone the repository
git clone https://github.com/<your-username>/WeatherSAM.git
cd WeatherSAM

# Create a virtual environment
conda create -n weathersam python=3.10 -y
conda activate weathersam

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the package
cd segment-anything
pip install -e .

# Install additional dependencies
pip install clip-by-openai geoclip scipy opencv-python-headless tqdm
```

### Download Pre-trained Weights / 下載預訓練權重

Place the following checkpoints under `segment-anything/checkpoints/`:

| File | Size | Source |
|---|---|---|
| `sam_vit_h_4b8939.pth` | ~2.4 GB | [SAM Official](https://github.com/facebookresearch/segment-anything) |
| `sam_vit_b_01ec64.pth` | ~358 MB | [SAM Official](https://github.com/facebookresearch/segment-anything) |
| `location_encoder_weights.pth` | ~37 MB | [GeoCLIP](https://github.com/VicenteVivan/geo-clip) |

---

## Dataset Preparation / 資料集準備

### Cityscapes-Foggy

```
data/cityscapes/
├── leftImg8bit_foggy/       # Foggy input images
├── leftImg8bit/             # Clear-weather reference images
└── gtFine/                  # Ground-truth semantic labels
```

### ACDC (Adverse Conditions Dataset with Correspondences)

```
data/acdc/
├── rgb_anon/                # Adverse weather input images (fog/rain/snow)
├── gt/                      # Ground-truth semantic labels
└── gt_ref_labelColor/       # Clear-weather reference color masks
```

### CSV Format / CSV 格式

**Cityscapes** (`train_with_gps.csv`):
```
image_path, ref_mask_path, gt_path, feature_path, lat, lon
```

**ACDC** (`acdc_train.csv`):
```
image_path, ref_mask_path, gt_path, feature_path, lat, lon, condition_id
```
- `condition_id`: 0=fog, 1=rain, 2=snow

---

## Training / 訓練

### Quick Start — Cityscapes / 快速開始（Cityscapes）

```bash
cd segment-anything
python train.py
```

預設參數已設定為 Cityscapes 模式，直接執行即可開始訓練。

### ACDC Fine-Tuning / ACDC 微調

```bash
python train.py \
    --checkpoint outputs_weather_sam_mask2former_testv1/weather_sam_best.pth \
    --train_csv /path/to/Datasets/acdc_train.csv \
    --val_csv /path/to/Datasets/acdc_val.csv \
    --use_condition_embedding \
    --epochs 30 \
    --lr 5e-5 \
    --output_dir outputs_acdc_finetune
```

### Key Hyperparameters / 關鍵超參數

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Total training epochs |
| `--batch_size` | 2 | Per-GPU batch size |
| `--accumulate_steps` | 4 | Gradient accumulation (effective batch = 8) |
| `--lr` | 5e-5 | Peak learning rate (cosine annealing + 5-epoch warmup) |
| `--focal_weight` | 4.0 | Focal loss weight for mask supervision |
| `--dice_weight` | 1.5 | Dice loss weight for mask supervision |
| `--ce_weight` | 1.0 | Context cross-entropy loss weight |
| `--abl_weight` | 0.5 | Active Boundary Loss weight |
| `--abl_start_epoch` | 35 | Epoch at which ABL activates |
| `--decoder_lr_scale` | 0.1 | Decoder token LR = main LR × 0.1 |
| `--transformer_lr_scale` | 0.01 | Transformer LR = main LR × 0.01 |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--max_norm` | 1.0 | Gradient clipping norm |
| `--use_condition_embedding` | False | Enable ACDC mode (ConditionEncoder) |

### Resume Training / 恢復訓練

```bash
python train.py --resume outputs_weather_sam_mask2former_testv1/weather_sam_best_latest.pth
```

### Trainable vs. Frozen Modules / 可訓練與凍結模組

| Module | Status | LR Scale | Role |
|---|---|---|---|
| `image_encoder` (ViT-H) | **Frozen** | — | Appearance feature extraction |
| `clip_model` | **Frozen** | — | Text semantic embedding |
| `location_encoder` backbone | **Frozen** | — | Geographic prior |
| `fusion_module` (CrossViewAlignment) | **Trainable** | 1× | Cross-frame geometric alignment |
| `gate_module` (GatedFusion) | **Trainable** | 1× | Adaptive weather blending |
| `context_fusion_head` (ResidualDWConvFusion) | **Trainable** | 1× | Pixel-level spatial refinement |
| `mask_encoder` | **Trainable** | 1× | Reference mask encoding |
| `text_encoder.projection` | **Trainable** | 1× | Text → SAM space projection |
| `location_encoder.output_projection` | **Trainable** | 1× | Geo → SAM space projection (Cityscapes) |
| `condition_encoder` | **Trainable** | 1× | Weather condition embedding (ACDC) |
| `mask_decoder.class_mask_tokens` | **Trainable** | 1× | 19 per-class query embeddings |
| `mask_decoder.class_hypernetworks_mlps` | **Trainable** | 1× | 19 per-class mask weight generators |
| `mask_decoder.output_upscaling` | **Trainable** | 1× | 64×64 → 256×256 feature upsampling |
| `mask_decoder.iou_token` / `mask_tokens` | **Trainable** | 0.1× | SAM original tokens (backward compat) |
| `mask_decoder.transformer` | **Trainable** | 0.01× | TwoWayTransformer (very low LR fine-tune) |
| `pe_layer` | **Trainable** | 1× | Positional encoding |

---

## Inference / 推論

> ⚠️ v2 的新架構無法使用 v1 的 checkpoint 進行推論。必須使用 v2 架構訓練後的 checkpoint。

```bash
cd segment-anything
python test_inference.py \
    --checkpoint outputs_weather_sam_mask2former_testv1/weather_sam_best.pth \
    --model_type vit_h \
    --image_path /path/to/foggy_image.png \
    --ref_mask_path /path/to/clear_reference_mask.png \
    --lat 48.8566 --lon 2.3522 \
    --output_dir inference_results/
```

---

## Results / 實驗結果

> Results will be updated upon completion of full training runs.
> 完整訓練完成後將更新結果。

| Method | Backbone | Decoder Style | mIoU (%) | Trainable Params |
|---|---|---|---|---|
| WeatherSAM v1 | ViT-H | Per-class independent (3 candidates) | — | ~10–20% |
| **WeatherSAM v2** | ViT-H | **Mask2Former-style unified query** | **—** | ~10–20% |

---

## Project Structure / 專案結構

```
WeatherSAM/
├── segment-anything/
│   ├── segment_anything/
│   │   ├── modeling/
│   │   │   ├── weather_sam.py              # Main model (Mask2Former-style forward)
│   │   │   ├── weather_mask_decoder.py     # forward_semantic() + predict_masks_semantic()
│   │   │   ├── fusion.py                   # CrossViewAlignment + GatedFusion
│   │   │   ├── fusion_head.py              # ResidualDWConvFusion (new) + ContextFusionHead (legacy)
│   │   │   ├── weather_prompt_encoder.py   # Multi-modal prompt encoder
│   │   │   ├── mask_encoder.py             # Reference mask encoder
│   │   │   ├── text_encoder.py             # CLIP text encoder wrapper
│   │   │   └── location_encoder.py         # GeoCLIP location encoder
│   │   └── build_weather_sam.py            # Model builder factory
│   ├── utils/
│   │   ├── new_loss.py                     # MaskLoss + ContextLoss + ABL (IoU MSE removed)
│   │   └── weather_dataloader.py           # Dataset with GPS/condition_id + reference mask
│   ├── train.py                            # Training entry point
│   ├── weather_trainer.py                  # Training loop (Mask2Former-style 6-stage)
│   ├── test_inference.py                   # Inference script
│   └── precompute_features.py              # ViT feature caching
├── Datasets/
│   ├── train_with_gps.csv                  # Cityscapes training manifest
│   ├── val_with_gps.csv                    # Cityscapes validation manifest
│   ├── acdc_train.csv                      # ACDC training manifest (with condition_id)
│   └── acdc_val.csv                        # ACDC validation manifest
├── README.md                               # v1 documentation
└── README_v2.md                            # v2 documentation (this file)
```

---

## Version Rollback / 版本回滾

若 v2 架構表現不如預期，可透過 git 回滾至 v1：

```bash
# 查看 v1 的最後一個 commit
git log --oneline

# 回滾所有檔案至 v1 狀態（commit hash 以實際為準）
git checkout 3a021e6 -- .

# 確認回到 v1 後重新訓練
python train.py
```

---

## Citation / 引用

```bibtex
@article{weathersam2025,
  title     = {WeatherSAM: Adverse Weather Semantic Segmentation via Cross-View Fusion and SAM Adaptation},
  author    = {[Author Names]},
  journal   = {[Venue]},
  year      = {2025},
  url       = {https://github.com/<your-username>/WeatherSAM}
}
```

We build upon the following foundational works:

```bibtex
@inproceedings{kirillov2023sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{cheng2022mask2former,
  title     = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author    = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and Kirillov, Alexander and Girdhar, Rohit},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}

@inproceedings{vivanco2024geoclip,
  title     = {Predicting Image Geolocalization via CLIP-Supervised Training},
  author    = {Vivanco, Vicente and others},
  booktitle = {CVPR},
  year      = {2024}
}
```

---

<div align="center">

*WeatherSAM v2 — Seeing Through the Fog, Together*

*WeatherSAM v2 — 穿透迷霧的視覺，眾類協作*

</div>
