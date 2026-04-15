# WeatherSAM: Adverse-Weather Semantic Segmentation via Mask2Former-Style Query Decoding and SAM Adaptation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](segment-anything/LICENSE)
[![Stars](https://img.shields.io/github/stars/your-org/WeatherSAM?style=social)](#)

</div>

---

## Abstract

Semantic segmentation under adverse weather (fog, rain, snow) remains challenging due to degraded appearance cues and domain shift from clear-weather training distributions. We present **WeatherSAM v2**, a parameter-efficient adaptation of the Segment Anything Model (SAM) that reformulates per-class mask prediction as a **Mask2Former-style unified query decoding** problem. The framework introduces nineteen class-specific learnable query tokens that are jointly processed in a single TwoWayTransformer forward pass, inducing mutual exclusivity through cross-class self-attention and eliminating the IoU-based candidate-selection mechanism used in prior SAM-based semantic variants. A clear-weather reference frame is encoded as a geometric prior and fused into the degraded feature map via cross-view attention and a gated blending module; pixel-level refinement is handled by a lightweight zero-initialized residual depthwise-separable block. For the ACDC benchmark, a learnable condition encoder replaces the geolocation prior. Approximately 10–20% of SAM's parameters are trained; the ViT-H backbone and CLIP text encoder are kept frozen.

**[中文摘要]** 本研究提出 **WeatherSAM v2**，一個基於 SAM 凍結骨幹的參數高效語意分割框架，針對惡劣天氣（霧、雨、雪）場景重構解碼範式。核心設計以 Mask2Former 風格的統一查詢機制取代原本各類別獨立解碼的流程：十九個類別專屬的可學習查詢 Token 在單次 TwoWayTransformer forward pass 中進行跨類別自注意力交互，自然形成互斥感知並移除 IoU 候選選擇的需求。晴天參考影像作為幾何先驗，經跨視角注意力與閘控融合模組注入至退化特徵圖；像素級精修則由輕量殘差深度可分離區塊完成。針對 ACDC 基準，以可學習的條件編碼器取代地理座標先驗。全程約僅訓練 10–20% 的 SAM 參數。

---

## News / Updates

- **[2026-04]** Architecture refactored to Mask2Former-style unified query decoding; IoU-prediction head and loss deprecated.
- **[2026-04]** ACDC dataset support added via `ConditionEncoder` and `--use_condition_embedding` flag.
- **[2026-03]** Active Boundary Loss (ABL) integrated with configurable warmup scheduling.
- **[2026-02]** Global mIoU evaluation protocol and ContextFusionHead (v14) introduced.
- **[2025-12]** Initial release: WeatherSAM v1 with per-class independent decoding and IoU candidate selection.

---

## Overview

![Architecture](assets/overview.png)

> *Figure placeholder. The pipeline comprises six stages: (1) multi-modal feature extraction, (2) cross-view fusion of clear-weather reference and adverse-weather features, (3) prompt encoding (text + location/condition), (4) Mask2Former-style decoding with K class-specific queries in a single Transformer pass, (5) residual depthwise-separable pixel refinement, and (6) multi-term loss computation.*

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-enabled NVIDIA GPU with ≥ 24 GB VRAM (validated on RTX 3090 / A100)
- PyTorch ≥ 2.0

### Environment Setup

```bash
# Clone repository
git clone https://github.com/<your-org>/WeatherSAM.git
cd WeatherSAM

# Create conda environment
conda create -n weathersam python=3.10 -y
conda activate weathersam

# Install PyTorch (adjust CUDA toolkit version as appropriate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM subpackage in editable mode
cd segment-anything
pip install -e .

# Install auxiliary dependencies
pip install clip-by-openai geoclip scipy opencv-python-headless tqdm pandas matplotlib
```

### Pre-trained Checkpoints

The following weights are required under `segment-anything/checkpoints/`:

| File | Size | Source |
|------|------|--------|
| `sam_vit_h_4b8939.pth` | ~2.4 GB | [Segment Anything (Kirillov et al., 2023)](https://github.com/facebookresearch/segment-anything) |
| `sam_vit_b_01ec64.pth` | ~358 MB | [Segment Anything (Kirillov et al., 2023)](https://github.com/facebookresearch/segment-anything) |
| `location_encoder_weights.pth` | ~37 MB | [GeoCLIP (Vivanco et al., 2024)](https://github.com/VicenteVivan/geo-clip) |

---

## Quick Start

A minimal Cityscapes-Foggy training run can be launched with default hyperparameters:

```bash
cd segment-anything
python train.py
```

Expected console output at initialization:

```
==========================================
🚀 WeatherSAM Training — Mask2Former-style
   Model: vit_h  |  Classes: 19
   Epochs: 100  |  Batch: 2 × 4 (accum) = 8 effective
   LR: 5e-5  |  Focal: 4.0  |  Dice: 1.5  |  CE: 1.0
   ABL weight: 0.5  (activates at epoch 35)
==========================================
```

---

## Dataset Preparation

### Cityscapes-Foggy

```
data/cityscapes/
├── leftImg8bit_foggy/    # Adverse-weather input images
├── leftImg8bit/          # Clear-weather reference images
└── gtFine/               # Ground-truth semantic labels (19 classes)
```

Manifest CSV (`train_with_gps.csv` / `val_with_gps.csv`):
```
image_path, ref_mask_path, gt_path, feature_path, lat, lon
```

### ACDC (Adverse Conditions Dataset with Correspondences)

```
data/acdc/
├── rgb_anon/             # Fog / rain / snow input images
├── gt/                   # Ground-truth semantic labels
└── gt_ref_labelColor/    # Clear-weather reference color masks
```

Manifest CSV (`acdc_train.csv` / `acdc_val.csv`):
```
image_path, ref_mask_path, gt_path, feature_path, lat, lon, condition_id
```
where `condition_id ∈ {0: fog, 1: rain, 2: snow}`.

### Feature Precomputation (optional)

ViT-H image embeddings may be cached offline to accelerate training:

```bash
python precompute_features.py --csv /path/to/train_with_gps.csv --output_dir /path/to/features/
```

---

## Training

### Stage 1 — Cityscapes-Foggy Pre-training

```bash
python train.py \
    --model_type vit_h \
    --train_csv /path/to/train_with_gps.csv \
    --val_csv   /path/to/val_with_gps.csv \
    --output_dir outputs_weather_sam_mask2former_testv1 \
    --epochs 100 --batch_size 2 --accumulate_steps 4 \
    --lr 5e-5 --focal_weight 4.0 --dice_weight 1.5 \
    --abl_weight 0.5 --abl_start_epoch 35
```

### Stage 2 — ACDC Fine-tuning

```bash
python train.py \
    --checkpoint outputs_weather_sam_mask2former_testv1/weather_sam_best.pth \
    --train_csv /path/to/acdc_train.csv \
    --val_csv   /path/to/acdc_val.csv \
    --use_condition_embedding \
    --epochs 30 --lr 5e-5 \
    --output_dir outputs_acdc_finetune
```

### Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Total training epochs |
| `--batch_size` | 2 | Per-GPU batch size |
| `--accumulate_steps` | 4 | Gradient accumulation (effective batch = 8) |
| `--lr` | 5e-5 | Peak learning rate; cosine annealing with 5-epoch linear warmup |
| `--focal_weight` | 4.0 | Per-class focal mask loss weight |
| `--dice_weight` | 1.5 | Per-class dice mask loss weight |
| `--ce_weight` | 1.0 | Global context cross-entropy weight |
| `--abl_weight` | 0.5 | Active Boundary Loss weight |
| `--abl_start_epoch` | 35 | Epoch index at which ABL is activated |
| `--decoder_lr_scale` | 0.1 | LR multiplier for legacy SAM tokens |
| `--transformer_lr_scale` | 0.01 | LR multiplier for the TwoWayTransformer |
| `--max_norm` | 1.0 | Gradient clipping L2 norm |
| `--use_condition_embedding` | False | Enables `ConditionEncoder` pathway (ACDC mode) |

### Trainable / Frozen Modules

| Module | Status | LR scale |
|--------|--------|----------|
| `image_encoder` (ViT-H) | Frozen | — |
| `clip_model` (text backbone) | Frozen | — |
| `location_encoder` (backbone) | Frozen | — |
| `fusion_module`, `gate_module` | Trainable | 1× |
| `context_fusion_head` (ResidualDWConvFusion) | Trainable | 1× |
| `mask_encoder`, text/location projections | Trainable | 1× |
| `condition_encoder` (ACDC) | Trainable | 1× |
| `mask_decoder.class_mask_tokens` | Trainable | 1× |
| `mask_decoder.class_hypernetworks_mlps` | Trainable | 1× |
| `mask_decoder.output_upscaling` | Trainable | 1× |
| `mask_decoder.{iou_token, mask_tokens}` | Trainable | 0.1× |
| `mask_decoder.transformer` | Trainable | 0.01× |

### Resuming a Run

```bash
python train.py --resume outputs_weather_sam_mask2former_testv1/weather_sam_best_latest.pth
```

---

## Evaluation

Semantic segmentation quality is reported as **mean Intersection-over-Union (mIoU)** over the 19 Cityscapes-compatible classes, computed globally (confusion matrix aggregated across the full validation split, not per-image averaged).

### Inference on a Single Sample

```bash
python test_inference.py \
    --checkpoint outputs_weather_sam_mask2former_testv1/weather_sam_best.pth \
    --model_type vit_h \
    --image_path    /path/to/foggy_image.png \
    --ref_mask_path /path/to/clear_reference_mask.png \
    --lat 48.8566 --lon 2.3522 \
    --output_dir inference_results/
```

> **Note.** Checkpoints produced by prior (v1) releases are not compatible with the Mask2Former-style decoder introduced in v2 due to modified token layout and the removal of the IoU prediction head.

### Results

Quantitative results on Cityscapes-Foggy and ACDC will be released upon completion of the full training protocol described above.

| Method | Backbone | Decoder | Cityscapes-Foggy mIoU | ACDC mIoU | Trainable Params |
|--------|----------|---------|-----------------------|-----------|------------------|
| WeatherSAM v1 | ViT-H | Per-class independent (3 candidates + IoU selection) | TBD | — | ~10–20% |
| **WeatherSAM v2 (ours)** | ViT-H | Mask2Former-style unified query | TBD | TBD | ~10–20% |

---

## Citation

If this work is used in academic research, please cite:

```bibtex
@article{weathersam2026,
  title   = {WeatherSAM: Adverse-Weather Semantic Segmentation via
             Mask2Former-Style Query Decoding and SAM Adaptation},
  author  = {[Author Names]},
  journal = {[Venue]},
  year    = {2026},
  note    = {Paper link TBD}
}
```

This framework builds upon the following foundational works:

```bibtex
@inproceedings{kirillov2023sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{cheng2022mask2former,
  title     = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author    = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and
               Kirillov, Alexander and Girdhar, Rohit},
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
  title     = {GeoCLIP: CLIP-Inspired Alignment between Locations and Images},
  author    = {Vivanco Cepeda, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle = {NeurIPS},
  year      = {2024}
}

@inproceedings{wang2022abl,
  title     = {Active Boundary Loss for Semantic Segmentation},
  author    = {Wang, Chi and Zhang, Yunke and Cui, Miaomiao and others},
  booktitle = {AAAI},
  year      = {2022}
}

@inproceedings{sakaridis2021acdc,
  title     = {{ACDC}: The Adverse Conditions Dataset with Correspondences for
               Semantic Driving Scene Understanding},
  author    = {Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  booktitle = {ICCV},
  year      = {2021}
}
```

---

## License

This repository inherits the **Apache License 2.0** from the underlying Segment Anything codebase. See [segment-anything/LICENSE](segment-anything/LICENSE) for the full text. Third-party components (CLIP, GeoCLIP) retain their respective licenses.

---

## Acknowledgements

This project builds directly on the open-source releases of **Segment Anything** (Meta AI Research), **Mask2Former** (Meta AI Research), **CLIP** (OpenAI), and **GeoCLIP**. We thank the authors of the Cityscapes, Foggy Cityscapes, and ACDC benchmarks for making their data publicly available.

Funding and institutional acknowledgements: *[To be added upon publication.]*

---

<div align="center">

*Paper link: TBD — [arXiv](#)*

</div>
