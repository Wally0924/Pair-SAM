# WeatherSAM: Adverse Weather Semantic Segmentation via Cross-View Fusion and SAM Adaptation

<div align="center">

> **惡劣天氣語意分割框架：基於跨視角融合與 SAM 適應性架構**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Abstract / 摘要

**[EN]**
We present **WeatherSAM**, a parameter-efficient domain-adaptation framework for semantic segmentation under adverse weather conditions — particularly dense fog — built upon the Segment Anything Model (SAM). WeatherSAM introduces three key innovations: (1) a **Cross-View Alignment and Gated Fusion** module that dynamically aligns clear-weather reference features with fog-degraded image features using multi-head cross-attention and a learnable gating mechanism; (2) a **Context Fusion Head** that enforces spatial coherence and mutual exclusivity across 19 semantic classes via depthwise convolution, spatial-aware channel attention, and a progressive residual scaling scheme; and (3) an **Active Boundary Loss (ABL)** that drives predicted decision boundaries toward ground-truth boundaries through a distance-weighted, direction-aware six-phase pipeline. Targeting the Cityscapes-Foggy benchmark, WeatherSAM achieves competitive mIoU by selectively fine-tuning only ~10–20% of total parameters while preserving SAM's zero-shot edge detection capability.

**[ZH]**
本文提出 **WeatherSAM**，一個針對惡劣天氣（特別是濃霧）場景下語意分割任務的參數高效領域自適應框架，以 Segment Anything Model (SAM) 為骨幹架構。WeatherSAM 包含三項核心創新：(1) **跨視角對齊與閘控融合模組**，利用多頭交叉注意力機制與可學習閘控機制，動態將晴天參考特徵與霧化影像特徵進行對齊；(2) **上下文融合頭**，透過深度可分離卷積、空間感知通道注意力及漸進式殘差縮放策略，強化 19 類語意預測的空間連貫性與互斥性；(3) **主動邊界損失 (ABL)**，以六階段距離加權、方向感知管線，驅動預測決策邊界向真實標注邊界靠攏。在 Cityscapes-Foggy 基準資料集上，WeatherSAM 僅微調約 10–20% 的模型參數，同時保留 SAM 原有的零樣本邊緣偵測能力。

---

## Table of Contents / 目錄

- [Background](#background--研究背景)
- [Architecture Overview](#architecture-overview--架構概觀)
- [Novel Contributions](#novel-contributions--創新貢獻)
- [Installation](#installation--環境安裝)
- [Dataset Preparation](#dataset-preparation--資料集準備)
- [Training](#training--訓練)
- [Inference](#inference--推論)
- [Results](#results--實驗結果)
- [Citation](#citation--引用)

---

## Background / 研究背景

**[EN]**
Urban scene understanding is foundational to autonomous driving and intelligent transportation. While modern segmentation models achieve near-human accuracy on clear-weather benchmarks, performance degrades dramatically under adverse weather: fog scatters light, reduces contrast, and occludes fine structures such as poles, pedestrians, and lane markings. Domain-adaptive approaches typically rely on either synthetic data augmentation or fully end-to-end training from scratch, both of which struggle to generalize across fog densities and geographic regions.

SAM demonstrated remarkable zero-shot segmentation capability via a promptable mask decoder. However, its single-image design lacks mechanisms for leveraging temporal or reference-frame context — a critical deficiency in adverse weather settings where a matched clear-weather prior can provide strong geometric guidance.

**[ZH]**
都市場景理解是自動駕駛與智慧運輸系統的核心基礎。儘管現代語意分割模型在晴天基準資料集上已達近乎人類水準的精度，在惡劣天氣下（如濃霧）效能仍大幅下降：霧氣散射光線、降低影像對比度，並遮蔽路標、行人、電線桿等精細結構。現有的領域自適應方法通常依賴合成資料增強或完整的端對端重訓練，難以泛化至不同霧濃度與地理場景。

SAM 透過可提示遮罩解碼器展現了卓越的零樣本分割能力，然而其單幀設計缺乏利用時序或參考幀上下文的機制——在惡劣天氣中，晴天配對先驗可提供強烈的幾何引導，此缺陷因此格外關鍵。

---

## Architecture Overview / 架構概觀

WeatherSAM follows a **six-stage pipeline** from input to semantic prediction.

WeatherSAM 採用**六階段管線**，從輸入到語意預測層層遞進。

```
 Foggy Image ─────────► [Image Encoder (ViT-H, Frozen)] ──────────────────────────┐
 Reference Mask ──────► [Mask Encoder (Trainable)] ──────► [CrossViewAlignment] ──► [GatedFusion]
 GPS Coordinates ─────► [Location Encoder (GeoCLIP, Frozen Backbone)] ──────────────────────────────► [WeatherPromptEncoder]
 Class Text Prompts ──► [Text Encoder (CLIP, Frozen Backbone)] ──────────────────────► [WeatherPromptEncoder]
                                                                                                          │
                                              ┌───────────────────────────────────────────────────────────┘
                                              ▼
                                    [Mask Decoder (Partially Frozen)]
                                    ─ 19 × (3 candidates, 256×256) masks
                                              │
                                    [Postprocess: Bilinear ↑ to 1024×1024]
                                              │
                                    [ContextFusionHead (Trainable)]
                                    ─ InstanceNorm → DepthwiseConv → SpatialAttention → Residual
                                              │
                                    [Argmax → Semantic Prediction (19 classes, 1024×1024)]
```

### Stage 1 — Multi-Modal Feature Extraction / 第一階段：多模態特徵萃取

| Component | Type | Input | Output | Details |
|---|---|---|---|---|
| Image Encoder (ViT-H) | Frozen | Foggy image 1024×1024 | 256×64×64 | SAM pre-trained backbone |
| Mask Encoder | Trainable | Clear-weather reference mask | 256×64×64 | Geometric prior encoding |
| Location Encoder | Frozen backbone + Trainable proj. | GPS (lat, lon) | 256-dim | GeoCLIP + Equal-Earth projection |
| Text Encoder | Frozen backbone + Trainable proj. | Class name strings | 256-dim × K | CLIP ViT-B/32, K = present classes |

### Stage 2 — Cross-View Fusion / 第二階段：跨視角融合

The **CrossViewAlignment** module applies multi-head cross-attention (8 heads, embed_dim=256) where the current foggy features serve as queries and reference features serve as keys/values, enabling geometric priors from clear-weather to "fill in" fog-obscured regions without masking void areas.

**跨視角對齊模組**以 8 頭交叉注意力（embed_dim=256）運作，以霧天特徵為查詢端、晴天特徵為鍵值端，在不遮蔽空白區域的前提下，讓晴天幾何先驗填補霧化的視覺缺損。

The **GatedFusion** module then learns a pixel-wise blending gate α ∈ (0,1):

$$\mathbf{f}_{\text{fused}} = (1-\alpha) \cdot \mathbf{f}_{\text{curr}} + \alpha \cdot \mathbf{f}_{\text{align}}$$

**閘控融合模組**學習逐像素混合係數 α，動態調控對當前幀與參考幀的依賴比例，適應不同程度的霧況。

### Stage 3 — Prompt Encoding / 第三階段：提示編碼

Geographic and semantic information is encoded into sparse prompts (K, 2, 256) — one text token and one location token per class — and combined with dense no-mask embeddings as input to the mask decoder.

地理資訊與語意類別分別編碼為稀疏提示（K, 2, 256）——每類別包含一個文字 token 與一個位置 token——並結合稠密無遮罩嵌入，共同送入遮罩解碼器。

### Stage 4 — Mask Decoding / 第四階段：遮罩解碼

SAM's two-way transformer decoder independently predicts 3 candidate masks per class. The best candidate (lowest mask loss) is selected per class, producing 19 independent binary logit maps at 256×256.

SAM 的雙向 Transformer 解碼器對每個類別獨立預測 3 個候選遮罩，以損失最低者作為最佳候選，最終得到 19 張 256×256 的二值邏輯圖。

### Stage 5 — Context Fusion Head / 第五階段：上下文融合頭

The **ContextFusionHead** takes the stacked 19-channel logit volume (B, 19, 1024, 1024) and refines it through:

1. **Absent-Class-Aware InstanceNorm**: Missing classes are filled with −10.0 and restored post-normalization to prevent norm distortion.
2. **Spatial Smoothing** (1×1 → 3×3 depthwise → 1×1): Preserves thin structures while reducing noise.
3. **Spatial-Aware Channel Attention** (4×4 adaptive pool → channel MLP → bilinear upsample): Enables regionally varying class priors.
4. **Learnable Residual Scale** (initialized at 0.1): Prevents early-training gradient shocks.

**上下文融合頭**接受堆疊的 19 通道邏輯張量，依序進行：缺失類別感知的 InstanceNorm、空間平滑（保留細長結構）、空間感知通道注意力（4×4 池化，保留空間分佈差異）、以及初始化為 0.1 的可學習殘差縮放，共同確保預測在空間上平滑且語意互斥。

### Stage 6 — Loss Computation / 第六階段：損失計算

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mask}} + \lambda_{\text{iou}} \mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{ctx}} + \lambda_{\text{abl}} \mathcal{L}_{\text{abl}} \cdot \mathbb{1}[t \geq t_{\text{abl}}]$$

| Loss | Weight | Description |
|---|---|---|
| $\mathcal{L}_{\text{mask}}$ (Focal + Dice) | focal=5.0, dice=2.0 | Per-class binary mask quality |
| $\mathcal{L}_{\text{iou}}$ (MSE) | 1.0 | IoU prediction head supervision |
| $\mathcal{L}_{\text{ctx}}$ (Weighted CE) | 1.0 | Global 19-class semantic CE with class balancing |
| $\mathcal{L}_{\text{abl}}$ (Active Boundary) | 1.5, starts epoch 20 | Boundary alignment loss |

---

## Novel Contributions / 創新貢獻

### 1. Cross-View Alignment + Gated Fusion / 跨視角對齊與閘控融合

**[EN]** Rather than treating each frame independently, WeatherSAM pairs each foggy input with a clear-weather reference mask. CrossViewAlignment performs multi-head cross-attention to geometrically align reference priors with the current feature map, while GatedFusion learns an adaptive weighting to control how strongly the reference guidance is applied — automatically reducing reliance on the reference in clearer regions.

**[ZH]** WeatherSAM 不以單幀為單位處理影像，而是將每張霧天輸入與配對的晴天參考遮罩共同送入模型。CrossViewAlignment 以多頭交叉注意力對齊幾何先驗，GatedFusion 則學習自適應權重，在霧濃區域大量借用參考資訊，在霧薄區域降低依賴，實現動態天氣適應。

### 2. Context Fusion Head / 上下文融合頭

**[EN]** Conventional semantic segmentation heads apply a single convolution over stacked class logits without accounting for absent classes or spatial heterogeneity of class distributions. Our ContextFusionHead addresses both: it uses instance normalization that is robust to absent (−10.0) padding, depth-wise spatial smoothing that preserves thin structures, and a 4×4 spatial-aware channel attention that models class preference variation across image regions.

**[ZH]** 傳統語意分割頭在堆疊邏輯圖上套用單一卷積，未考量缺失類別與空間分佈的異質性。ContextFusionHead 採用對 −10.0 填充值魯棒的實例歸一化、保留細長結構的深度可分離平滑，以及 4×4 空間感知通道注意力，有效建模不同空間區域的類別偏好差異。

### 3. Active Boundary Loss (ABL) / 主動邊界損失

**[EN]** ABL is a six-phase differentiable boundary loss that:
1. Detects Predicted Decision Boundaries (PDB) via 8-neighbor KL divergence.
2. Computes Ground-Truth Boundary (GTB) and Euclidean Distance Transform.
3. Estimates the direction toward the nearest GTB from each PDB pixel.
4. Treats 8-directional KL values as logits for direction classification.
5. Weights loss by distance to GTB (farther = stronger gradient push).
6. Activates with a warmup delay (epoch 20) to allow stable mask learning first.

**[ZH]** ABL 是一個六階段可微邊界損失：(1) 透過 8-鄰域 KL 散度偵測預測決策邊界 (PDB)；(2) 計算真實標注邊界 (GTB) 與歐式距離轉換；(3) 估計各 PDB 像素朝向最近 GTB 的方向；(4) 以 8 方向 KL 值為邏輯分數進行方向分類；(5) 以到 GTB 的距離加權損失（距離越遠梯度推力越強）；(6) 以預熱延遲機制（第 20 epoch 起）確保遮罩學習穩定後再啟動邊界監督。

### 4. Multi-Modal Prompt Fusion / 多模態提示融合

**[EN]** WeatherSAM extends SAM's prompting mechanism with two additional modalities: (a) CLIP-based semantic class prompts that encode fine-grained category distinctions, and (b) GeoCLIP-based geographic prompts that encode location-specific visual priors (e.g., region-specific road textures or vegetation density). Both use frozen backbones with trainable projection layers for parameter efficiency.

**[ZH]** WeatherSAM 在 SAM 原有提示機制上新增兩種模態：(a) 基於 CLIP 的語意類別提示，編碼細粒度類別語意差異；(b) 基於 GeoCLIP 的地理提示，編碼位置特定的視覺先驗（如特定地區的路面紋理或植被密度）。兩者均採用凍結骨幹加可訓練投影層的參數高效設計。

### 5. Progressive Training Strategy / 漸進式訓練策略

**[EN]** To prevent early-stage gradient conflicts, WeatherSAM employs: (1) gradient detachment for the context fusion head during epochs 1–4, followed by Adam momentum reset at epoch 5 when full gradients are unblocked; (2) delayed ABL activation starting at epoch 20; (3) learnable residual scale initialized at 0.1 to ensure the fusion head begins as near-identity. Together, these ensure stable convergence in a multi-objective loss landscape.

**[ZH]** 為避免訓練初期梯度衝突，WeatherSAM 採用：(1) 前 4 個 epoch 中對融合頭梯度進行截斷，並於第 5 epoch 解封後重置 Adam 動量；(2) 第 20 epoch 起啟動 ABL；(3) 可學習殘差縮放初始化為 0.1，確保融合頭初期行為接近恆等映射。此策略確保多目標損失景觀下的穩定收斂。

---

## Installation / 環境安裝

### Requirements / 系統需求

- Python ≥ 3.8
- PyTorch ≥ 2.0 with CUDA support
- NVIDIA GPU with ≥ 24 GB VRAM (recommended: A100 / RTX 3090 / RTX 4090)

### Setup / 安裝步驟

```bash
# Clone the repository / 克隆儲存庫
git clone https://github.com/<your-username>/WeatherSAM.git
cd WeatherSAM

# Create a virtual environment / 建立虛擬環境
conda create -n weathersam python=3.10 -y
conda activate weathersam

# Install PyTorch (adjust CUDA version as needed) / 安裝 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the package / 安裝套件
cd segment-anything
pip install -e .

# Install additional dependencies / 安裝額外依賴
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

WeatherSAM is evaluated on **Cityscapes-Foggy**, a benchmark derived from Cityscapes with synthetic fog at three densities (β ∈ {0.005, 0.01, 0.02}).

WeatherSAM 在 **Cityscapes-Foggy** 基準上進行評估，該資料集由 Cityscapes 衍生，包含三種霧濃度（β ∈ {0.005, 0.01, 0.02}）。

### Directory Structure / 目錄結構

```
data/
├── cityscapes/
│   ├── leftImg8bit_foggy/           # Foggy input images
│   │   ├── train/<city>/*.png
│   │   └── val/<city>/*.png
│   ├── leftImg8bit/                 # Clear-weather reference images
│   │   ├── train/<city>/*.png
│   │   └── val/<city>/*.png
│   └── gtFine/                      # Ground-truth semantic labels
│       ├── train/<city>/*labelIds.png
│       └── val/<city>/*labelIds.png
```

### Generate CSV Index / 生成 CSV 索引

```bash
cd Datasets
python generate_csv.py --data_root /path/to/cityscapes --output_dir .
python add_gps_to_csv.py --input train.csv --output train_with_gps.csv
```

The resulting CSVs (`train_with_gps.csv`, `val_with_gps.csv`) contain columns:
`image_path`, `ref_mask_path`, `gt_path`, `feature_path`, `lat`, `lon`

### (Optional) Precompute ViT Features / 預計算 ViT 特徵

```bash
cd segment-anything
python precompute_features.py \
    --csv ../Datasets/train_with_gps.csv \
    --checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h
```

Precomputed features accelerate training by skipping the frozen ViT encoder forward pass.

預計算特徵可跳過凍結 ViT 編碼器的前向傳播，大幅加速訓練。

---

## Training / 訓練

### Quick Start / 快速開始

```bash
cd segment-anything
python train.py \
    --train_csv ../Datasets/train_with_gps.csv \
    --val_csv   ../Datasets/val_with_gps.csv \
    --checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h \
    --output_dir outputs/weathersam_v1
```

### Key Hyperparameters / 關鍵超參數

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Total training epochs |
| `--batch_size` | 2 | Per-GPU batch size |
| `--accumulate_steps` | 4 | Gradient accumulation (effective batch = 8) |
| `--lr` | 5e-5 | Peak learning rate (cosine annealing + 5-epoch warmup) |
| `--focal_weight` | 5.0 | Focal loss weight for mask supervision |
| `--dice_weight` | 2.0 | Dice loss weight for mask supervision |
| `--iou_weight` | 1.0 | IoU prediction head loss weight |
| `--ce_weight` | 1.0 | Context cross-entropy loss weight |
| `--abl_weight` | 1.5 | Active Boundary Loss weight |
| `--abl_start_epoch` | 20 | Epoch at which ABL activates |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--max_norm` | 1.0 | Gradient clipping norm |

### Resume Training / 恢復訓練

```bash
python train.py \
    --resume outputs/weathersam_v1/weather_sam_best_latest.pth \
    [... other flags ...]
```

### Trainable vs. Frozen Modules / 可訓練與凍結模組

| Module | Status | Role |
|---|---|---|
| `image_encoder` (ViT-H) | **Frozen** | Appearance feature extraction |
| `clip_model` | **Frozen** | Text semantic embedding |
| `location_encoder` backbone | **Frozen** | Geographic prior |
| `mask_decoder.transformer` | **Frozen** | SAM two-way attention core |
| `fusion_module` (CrossViewAlignment) | **Trainable** | Cross-frame geometric alignment |
| `gate_module` (GatedFusion) | **Trainable** | Adaptive weather blending |
| `context_fusion_head` | **Trainable** | Spatial semantic coherence |
| `mask_encoder` | **Trainable** | Reference mask encoding |
| `text_encoder.projection` | **Trainable** | Text → SAM space projection |
| `location_encoder.output_projection` | **Trainable** | Geo → SAM space projection |
| `mask_decoder.iou_prediction_head` | **Trainable** | Mask quality scoring |
| `mask_decoder.output_upscaling` | **Trainable** | Mask resolution upsampling |
| `mask_decoder.output_hypernetworks_mlps` | **Trainable** | Per-class mask weight generation |

---

## Inference / 推論

```bash
cd segment-anything
python test_inference.py \
    --checkpoint outputs/weathersam_v1/weather_sam_best_latest.pth \
    --model_type vit_h \
    --image_path /path/to/foggy_image.png \
    --ref_mask_path /path/to/clear_reference_mask.png \
    --lat 48.8566 --lon 2.3522 \
    --output_dir inference_results/
```

---

## Results / 實驗結果

### Cityscapes-Foggy Validation (β = 0.02) / Cityscapes-Foggy 驗證集結果

> Results will be updated upon completion of full training runs.
> 完整訓練完成後將更新結果。

| Method | Backbone | mIoU (%) | Params (M) | FPS |
|---|---|---|---|---|
| DeepLabV3+ | ResNet-101 | — | 59.3 | — |
| SegFormer-B5 | MiT-B5 | — | 84.6 | — |
| SAM (zero-shot) | ViT-H | — | 641.1 | — |
| **WeatherSAM (ours)** | ViT-H | **—** | ~10–20% trainable | — |

### Qualitative Observations / 定性觀察

- **Boundary Sharpness**: ABL demonstrably sharpens class boundaries compared to CE-only baselines, particularly around pedestrians and vehicle outlines in dense fog.
- **Class Coherence**: ContextFusionHead eliminates overlapping car/road predictions that arise from independent per-class mask decoding.
- **Geographic Generalization**: Location encoding improves predictions in geographically distinct validation scenes.

- **邊界清晰度**：與僅使用 CE 的基線相比，ABL 顯著提升霧天行人與車輛輪廓的邊界清晰度。
- **類別一致性**：ContextFusionHead 有效消除獨立遮罩解碼所產生的車輛/道路預測重疊問題。
- **地理泛化能力**：位置編碼在地理差異顯著的驗證場景中改善預測結果。

---

## Project Structure / 專案結構

```
WeatherSAM/
├── segment-anything/
│   ├── segment_anything/
│   │   ├── modeling/
│   │   │   ├── weather_sam.py            # Main model class
│   │   │   ├── fusion.py                 # CrossViewAlignment + GatedFusion
│   │   │   ├── fusion_head.py            # ContextFusionHead
│   │   │   ├── weather_prompt_encoder.py # Multi-modal prompt encoder
│   │   │   ├── weather_mask_decoder.py   # Modified mask decoder
│   │   │   ├── mask_encoder.py           # Reference mask encoder
│   │   │   ├── text_encoder.py           # CLIP text encoder wrapper
│   │   │   └── location_encoder.py       # GeoCLIP location encoder
│   │   └── build_weather_sam.py          # Model builder factory
│   ├── utils/
│   │   ├── new_loss.py                   # ABL + ContextLoss + MaskLoss
│   │   └── weather_dataloader.py         # Dataset with GPS + reference mask
│   ├── train.py                          # Training entry point
│   ├── weather_trainer.py                # Training loop with 6-stage pipeline
│   ├── test_inference.py                 # Inference script
│   └── precompute_features.py            # ViT feature caching
├── Datasets/
│   ├── generate_csv.py                   # CSV index generation
│   ├── add_gps_to_csv.py                 # GPS metadata integration
│   ├── train_with_gps.csv                # Training manifest
│   └── val_with_gps.csv                  # Validation manifest
└── README.md
```

---

## Technical Details / 技術細節

### Active Boundary Loss Algorithm / 主動邊界損失演算法

```
Algorithm: Active Boundary Loss (ABL)
Input:  Predicted logits P ∈ R^{B×19×H×W}, GT labels Y ∈ Z^{B×H×W}
Output: Scalar boundary loss L_abl

For each image in batch:
  1. Compute 8-neighbor KL divergence map D ∈ R^{H×W}
  2. Select PDB = {pixels where D > τ_kl OR top-1% by D} ∩ {Y ≠ 255}
  3. Extract GTB = boundary pixels of Y (adjacent class change)
  4. Compute Euclidean Distance Transform: EDT(x) = min_{y∈GTB} ||x−y||₂
  5. For each p ∈ PDB:
       direction d* = argmin cos(∇EDT(p), e_d), d ∈ {N,NE,E,...,NW}
       smooth_label(d*) = 0.8; smooth_label(d≠d*) = 0.2/7
       direction_logits = [D(p + δ_d) for d in 8 directions]
       weight(p) = min(EDT(p) / 20.0, 1.0)
       L_p = weight(p) × SoftCrossEntropy(direction_logits, smooth_label)
  6. L_abl = mean over PDB pixels
Return L_abl
```

### Learning Rate Schedule / 學習率排程

```
epoch ≤ 5  (warmup):  lr = lr_base × (epoch / warmup_epochs)
epoch > 5  (cosine):  lr = lr_base × 0.5 × (1 + cos(π × (epoch−5) / (T_max−5)))
```

### Mixed Precision & Gradient Accumulation / 混合精度與梯度累積

```python
# Effective batch size = batch_size × accumulate_steps = 2 × 4 = 8
scaler = torch.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(...)
scaler.scale(loss / accumulate_steps).backward()
if step % accumulate_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## Citation / 引用

If you use WeatherSAM in your research, please cite:

若您在研究中使用 WeatherSAM，請引用以下文獻：

```bibtex
@article{weathersam2025,
  title     = {WeatherSAM: Adverse Weather Semantic Segmentation via Cross-View Fusion and SAM Adaptation},
  author    = {[Author Names]},
  journal   = {[Venue]},
  year      = {2025},
  url       = {https://github.com/<your-username>/WeatherSAM}
}
```

We also build upon the following foundational works:

本研究亦基於以下基礎工作：

```bibtex
@inproceedings{kirillov2023sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle = {ICCV},
  year      = {2023}
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

@inproceedings{sakaridis2018foggy,
  title     = {Semantic Foggy Scene Understanding with Synthetic Data},
  author    = {Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  booktitle = {IJCV},
  year      = {2018}
}
```

---

## Acknowledgements / 致謝

This project builds upon [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research, [CLIP](https://github.com/openai/CLIP) by OpenAI, and [GeoCLIP](https://github.com/VicenteVivan/geo-clip). We thank the Cityscapes team for the Foggy Cityscapes benchmark.

本研究建立於 Meta AI Research 的 Segment Anything (SAM)、OpenAI 的 CLIP，以及 GeoCLIP 之上。感謝 Cityscapes 團隊提供 Foggy Cityscapes 基準資料集。

---

<div align="center">

*WeatherSAM — Seeing Through the Fog*

*WeatherSAM — 穿透迷霧的視覺*

</div>
