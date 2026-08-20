# PairSAM: Reference-Guided Adaptation of SAM for Adverse-Weather Semantic Segmentation

<div align="center">

**惡劣天氣語意分割：以晴天參考影像引導的 SAM 參數高效適應框架**

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](segment-anything/LICENSE)

[English](README.md) | 繁體中文

</div>

---

## 摘要

語意分割在惡劣天氣（霧、雨、雪、夜間）下劇烈退化：對比度下降與紋理破壞削弱了晴天模型賴以判斷的外觀線索。本研究提出 **PairSAM**，一個參數高效框架，以凍結的 Segment Anything Model（SAM）ViT-H 骨幹為基礎，透過注入「幾何對齊後的晴天參考影像」作為結構先驗來適應退化場景。GNSS 配對的晴天影像先由凍結的光流對齊網路扭曲至惡劣天氣視角，其多尺度 VGG 特徵再經對齊信心遮罩閘控，透過多尺度交叉注意力 Adapter 注入 ViT-H 編碼器。解碼採 Mask2Former 風格的**統一查詢**（unified query）形式：十九個類別專屬查詢 Token 在單次 TwoWayTransformer 中經跨類別自注意力交互；最後以輕量殘差深度卷積頭對組裝後的類別 logit 圖進行跨類別競爭與空間精修。全程僅訓練約 **2.98%（24.53M）** 參數，ViT-H 影像編碼器、CLIP 文字編碼器與對齊網路皆凍結。實驗於 ACDC、Dark Zurich、RobotCar Correspondence 三個資料集、依標準 GNSS 配對協定進行。

---

## 簡介

駕駛場景分割恰恰在最需要它的時刻失效。濃霧中對比度崩塌，遠處車流融成一片灰；夜間無照明的人行道與騎士消失於黑暗，車燈卻使感測器飽和；雨雪破壞了區分道路與地形的局部紋理。以晴天影像訓練的模型把這些外觀線索當作主要判斷依據，線索一旦消失，精度便急遽下滑。SAM 這類視覺基礎模型帶來十億級遮罩規模學得的分割先驗，然而 SAM 的訓練目標是類別無關、提示驅動的遮罩生成，本身不輸出語意標籤，更遑論在退化影像上。

既有兩條研究路線各解決了問題的一部分。參數高效的 SAM 適應方法以輕量 Adapter 或重訓解碼器讓凍結骨幹特化至下游領域；然而它們把退化影像當作唯一觀測，在觀測本身不可靠時沒有任何外部結構可用。反之，Refign 與 CMA 證明了同一場景的 GNSS 配對晴天影像是惡劣天氣分割的有力引導；但兩者皆微調整個分割網路，使配對影像先驗與全網重訓綁定，也放棄了基礎模型凍結特徵的優勢。配對晴天影像所提供的結構先驗，與凍結基礎模型適應的高效率，至今尚未被結合。

PairSAM 將兩者結合：幾何對齊、信心閘控的晴天參考被注入凍結的 SAM ViT-H 編碼器，注入後的特徵在單次統一查詢解碼中輸出十九類語意圖，全程訓練參數不到模型的 3%。本研究的貢獻如下：

- **參考注入（Reference injection）。** 凍結的光流網路將 GNSS 配對晴天影像扭曲至惡劣天氣視角；其多尺度 VGG 特徵經逐像素對齊信心遮罩閘控後，由 **WarpedVGG Adapter**（多尺度交叉注意力模組，各階段閘門可學習）注入 ViT-H 第 7/15/23/31 個 block。不可靠的對應關係在誤導解碼器之前即被抑制。
- **統一查詢解碼（Unified query decoding）。** 十九個類別專屬查詢在單次 TwoWayTransformer 中聯合解碼，跨類別自注意力誘導互斥性，並移除既有 SAM 語意變體的 IoU 候選挑選步驟；隨後由殘差深度卷積頭（**LRH**）對組裝後的 logit 圖進行跨類別競爭與空間精修。
- **參數效率。** 僅訓練 24.53M 參數（全模型的 2.98%）；ViT-H 影像編碼器、CLIP 文字編碼器與 VGG+光流對齊網路皆凍結。每項架構選擇均對應單一 CLI 旗標，可進行受控的單變因消融。

---

## 方法總覽

<p align="center"><em>[圖示佔位 — pipeline 示意圖待補]</em></p>

PairSAM 將一張惡劣天氣影像與其 GNSS 配對晴天參考,經以下階段映射為 19 類語意圖。

1. **參考對齊（凍結）。** `CMAAlignment`（VGG-16 + UAWarpC）估計晴天參考至惡劣天氣視角的光流,扭曲參考特徵,並產生對齊信心圖。此模組載入預訓練權重後保持凍結。
2. **參考注入。** `MultiScaleCrossAttnInjector`（即 *WarpedVGG Adapter*）以多尺度交叉注意力將信心閘控後的參考特徵注入 ViT-H 編碼器。注入位置為第 7/15/23/31 個 block 的自注意力**之前**（pre），各階段閘門可學習。
3. **提示編碼。** 類別名稱文字嵌入（凍結 CLIP ViT-B/32 + 可訓練投影）與可學習的**條件嵌入**（fog/rain/snow/night）構成稀疏提示;ACDC 以條件編碼器取代地理位置先驗。
4. **統一遮罩解碼。** `MaskDecoder` 將所有啟用的類別查詢置於同一 TwoWayTransformer 序列（跨類別自注意力）;每個類別有專屬的 hypernetwork MLP 產生其動態遮罩頭。每類一張遮罩,無 IoU 候選挑選。
5. **Logit 精修（LRH）。** `ResidualDWConvFusion` 對組裝後的 `(1, 19, H, W)` logit 圖先做跨類別競爭（1×1 mixer、殘差）,再做深度卷積空間平滑。
6. **損失函數。** 加權交叉熵（**median-frequency balancing, MFB**）+ **Lovász-Softmax** + **Dice**。

### 可訓練 / 凍結模組

| 模組 | 狀態 | 參數量 |
|---|---|---|
| SAM ViT-H 影像編碼器 | 凍結 | 637.03 M |
| CLIP 文字編碼器（骨幹） | 凍結 | 151.28 M |
| CMAAlignment（VGG-16 + UAWarpC） | 凍結 | 10.78 M |
| `MultiScaleCrossAttnInjector`（WarpedVGG Adapter） | **可訓練** | 17.32 M |
| `TwoWayTransformer` | **可訓練** | 3.29 M |
| `class_hypernetworks_mlps` | **可訓練** | 2.66 M |
| `pe_layer` | **可訓練** | 1.05 M |
| 文字投影 | **可訓練** | 0.13 M |
| `output_upscaling` | **可訓練** | 0.07 M |
| `class_mask_tokens` | **可訓練** | 4.9 K |
| `ResidualDWConvFusion`（LRH） | **可訓練** | 2.9 K |
| `ConditionEncoder` | **可訓練** | 1.0 K |
| **可訓練總計** | | **24.53 M（2.98%）** |

---

## 實驗數據

所有數字依循 Refign 與 CMA 的 GNSS 配對評估協定：19 個 Cityscapes 類別的 mIoU,於原生標註解析度（1080×1920）計算,混淆矩陣對整個 split 全域彙總。

> 量化結果將於多 seed 訓練協定完成後釋出,下表數字為佔位。

### ACDC — 整體與各天候 mIoU（驗證集,406 張）

| 方法 | 骨幹 | 參考 | 霧 | 雨 | 雪 | 夜 | mIoU |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| CMA | SegFormer | ✓ | — | — | — | — | — |
| Refign-DAFormer | MiT-B5 | ✓ | — | — | — | — | — |
| **PairSAM（本研究）** | ViT-H（凍結） | ✓ | TBD | TBD | TBD | TBD | **TBD** |

### 跨資料集泛化（mIoU）

| 方法 | Dark Zurich（test） | RobotCar Corr.（test） |
|---|:--:|:--:|
| CMA | 53.6 | 54.3 |
| Refign-DAFormer | 56.2 | 60.5 |
| **PairSAM（本研究）** | **TBD** | **TBD** |

> ACDC **test set** 的逐類 IoU 與既有 Cityscapes→ACDC 方法之比較（CMA Table-1 格式）於論文中報告;預測結果以 `scripts/eval/dump_acdc_test_preds.py` 匯出後提交官方 server。

---

## 安裝

### 環境需求

- Python ≥ 3.10
- NVIDIA GPU,VRAM ≥ 24 GB（於 RTX 4090 驗證）
- PyTorch 2.x（CUDA 版本）

### 安裝步驟

```bash
git clone https://github.com/Wally0924/Pair-SAM.git
cd Pair-SAM

conda create -n sam_env python=3.10 -y
conda activate sam_env

pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
cd segment-anything
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install -e .
```

實測環境:Python 3.10 / Ubuntu 24.04 / CUDA 12.8 / RTX 4090（24 GB）;此環境下 `python -m pytest tests/ -q` 全數通過。若 CUDA toolkit 版本不同,請自行調整 PyTorch 的 index URL。

### 預訓練權重

模型權重**不存放於本 repo**。請自行下載並放置於 `segment-anything/checkpoints/`:

| 檔案 | 大小 | 用途 | 來源 |
|---|---|---|---|
| `sam_vit_h_4b8939.pth` | ~2.4 GB | 所有 ViT-H 實驗 | [Segment Anything (Kirillov et al., ICCV 2023)](https://github.com/facebookresearch/segment-anything) |
| `sam_vit_b_01ec64.pth` | ~358 MB | ViT-B 消融 | [Segment Anything](https://github.com/facebookresearch/segment-anything) |
| `cma_alignment_weights.pth` | ~69 MB | CMA 對齊（VGG-16 + UAWarpC） | *[下載連結待補]* |
| `cma_segformer_acdc.ckpt` | ~1.4 GB | CMA 基線比較 | *[下載連結待補]* |
| `location_encoder_weights.pth` | ~37 MB | GNSS 位置編碼 | *[下載連結待補]* |
| `cityscapes_pretrain/sam_vit_h_cityscapes_merged.pth` | ~2.4 GB | 晴天預訓練階段 | *[下載連結待補]* |

### 訓練完成的模型與實驗紀錄

訓練輸出寫入 `segment-anything/outputs_*/`,不納入版本控制（僅 checkpoint 即超過 100 GB）。訓練完成的 PairSAM 權重與對應實驗紀錄 — `train_log.csv`、`e1_results.json`、`ablation_config.json` 及各消融變體的訓練曲線 — 另行釋出:*[下載連結待補]*。

---

## 資料集

PairSAM 依循與 Refign、CMA 相同的 **GNSS 配對**協定評估:每張惡劣天氣目標影像都帶有一張以 GNSS 檢索的晴天參考,作為結構先驗。所有資料集皆採 19 個 Cityscapes 訓練標籤類別。

| 資料集 | Train | Val | Test | 主要變因 |
|---|---:|---:|---:|---|
| **ACDC** | 1,600 | 406 | 2,000 | 霧 / 雨 / 雪 / 夜 |
| **Dark Zurich** | 2,416 | 50 | 151 | 夜間 |
| **RobotCar Correspondence** | 6,511 | 27 | 27 | 跨季節 / 跨時段 |

本 repo **不散布資料集** — 請依各自授權向官方來源取得（ACDC、MUSES、Cityscapes、Foggy Cityscapes、Dark Zurich）,保持原始目錄結構,放置於同一根目錄下。連結與預期結構見 [`Datasets/README.md`](Datasets/README.md)。

Manifest CSV 欄位（`acdc_adverse_ref_rgb_{train,val}.csv`）:

```text
image_path, ref_image_path, gt_path, condition, condition_id, invalid_mask
```
其中 `condition_id ∈ {0: fog, 1: rain, 2: snow, 3: night}`。

### 路徑設定

Manifest CSV 以佔位符取代絕對路徑,執行前請先設定資料集根目錄:

```bash
export DATASET_ROOT=/path/to/your/Datasets   # 預設 ~/Datasets
```

`${DATASET_ROOT}`（外部資料集）與 `${REPO_ROOT}`（repo 內特徵快取）由 [`Datasets/path_resolver.py`](Datasets/path_resolver.py) 在 `pair_dataloader.py`、`precompute_features.py` 或 `train.py` 讀取 manifest 時自動展開,無需手動替換。

> **注意。** `segment-anything/scripts/` 下的部分輔助腳本（如 `run_muses_cond8.sh`、`make_ch4_figs.py`）的 argparse 或 shell 預設值仍保留原作者的絕對路徑。這些是一次性分析與繪圖工具 — 執行前請覆寫相關的 `--csv` / `--output_dir` 引數或修改預設值。本 README 記載的訓練與評估進入點皆接受明確路徑,無需修改。

<!-- -->

> **注意。** ACDC test set 的 GT 由官方 server 保留;本 repo 的逐天候與消融分析使用標籤公開的 ACDC **驗證集**（406 張）。

---

## 訓練

完整模型（**FULL**）對應 `train.py` 預設值:參考注入開啟、pre-block 注入、統一解碼、精修頭開啟,目標函數為 CE+Lovász+Dice+MFB。

```bash
cd segment-anything
python train.py \
  --model_type vit_h \
  --train_csv /path/to/acdc_adverse_ref_rgb_train.csv \
  --val_csv   /path/to/acdc_adverse_ref_rgb_val.csv \
  --inject pre --decoder unified --lrh --mfb \
  --lovasz_weight 1 --dice_weight 1 \
  --epochs 50 --patience 10 --batch_size 1 --accumulate_steps 4 --lr 5e-5 \
  --seed 42 --output_dir outputs/full_seed42
```

### 主要超參數

| 引數 | 預設 | 說明 |
|---|---|---|
| `--lr` | 5e-5 | 峰值學習率;5 epoch 線性 warmup + cosine 衰減 |
| `--epochs` | 30 | 上限;以 val mIoU 早停（`--patience 10`） |
| `--batch_size` / `--accumulate_steps` | 1 / 4 | 有效 batch 4（梯度累積,AMP） |
| `--lovasz_weight` / `--dice_weight` | 1.0 / 1.0 | Lovász-Softmax / Dice 權重（皆設 0 → 純 CE） |
| `--adapter_lr_scale` | 3.0 | WarpedVGG Adapter 的學習率倍率 |
| `--decoder_lr_scale` / `--transformer_lr_scale` | 0.5 / 0.05 | 解碼器 token / transformer 的學習率倍率 |

### 消融開關

每個設計維度對應單一旗標,可在固定其餘設定下切換任一元件:

| 旗標 | 效果 |
|---|---|
| `--inject {pre,post}` | 參考注入於 ViT block 自注意力之前 / 之後 |
| `--decoder {unified,per_class}` | 聯合解碼 vs. 逐類獨立解碼（參數量相同） |
| `--lrh / --no-lrh` | 開啟 / 關閉殘差精修頭 |
| `--mfb / --no-mfb` | Median-frequency 類別加權 vs. 均勻加權 |
| `--ref / --no-ref` | 注入參考內容 vs. 歸零（分離參考訊號與 Adapter 容量） |
| `--use_vgg_adapter / --no-use_vgg_adapter` | 完全開啟 / 關閉 Adapter |

> **關於 Rare-Class Sampling（RCS）。** 已實作 DAFormer 風格的稀有類別資料取樣器（`--rcs`,預設關閉）,並與損失層級的 MFB 平衡比較。在本設定下單用 MFB 較優,故 RCS **不在**釋出的 FULL 配置中;程式碼為重現性而保留,MFB 與 RCS 的比較記載於釋出的實驗紀錄。

---

## 評估

mIoU 於 19 類、原生標註解析度（1080×1920）計算,混淆矩陣對整個 split 全域彙總（與 ACDC server 及 Refign/CMA 協定一致）。

```bash
# 逐類 × 逐天候 IoU + 整體 / 逐天候 mIoU（JSON + Markdown）
python scripts/eval/eval_e1_acdc_val_full.py \
  --ckpt outputs/full_seed42/weather_sam_best_latest.pth \
  --out  outputs/full_seed42/e1_results.json
```

---

## 專案結構

```
Pair-SAM/
├── segment-anything/
│   ├── segment_anything/modeling/
│   │   ├── pair_sam.py               # 頂層模型;參考預對齊 + Adapter
│   │   ├── vgg_adapter.py            # MultiScaleCrossAttnInjector（WarpedVGG Adapter）
│   │   ├── fusion.py                 # CMAAlignment（VGG-16 + UAWarpC）+ 光流引導融合
│   │   ├── fusion_head.py            # ResidualDWConvFusion（LRH）
│   │   ├── pair_mask_decoder.py      # Mask2Former 風格統一 / 逐類解碼器
│   │   ├── pair_prompt_encoder.py    # 文字 + 條件提示編碼器
│   │   ├── transformer.py            # TwoWayTransformer
│   │   └── semantic_assembly.py      # 19 類 logit 組裝 + 閘控 LRH
│   ├── utils/
│   │   ├── new_loss.py               # ContextLoss（CE+MFB+Lovász）+ MaskLoss（Dice）
│   │   ├── rare_class_sampler.py     # DAFormer 風格 RCS（選用,預設關閉）
│   │   └── pair_dataloader.py        # ACDC 資料集（惡劣天氣 + 參考 + 條件）
│   ├── scripts/
│   │   ├── eval/eval_e1_acdc_val_full.py   # 逐類 × 逐天候 mIoU
│   │   └── aggregate_ablation.py           # 消融表 → LaTeX
│   ├── train.py                      # 訓練進入點
│   ├── pair_trainer.py               # 訓練 / 驗證迴圈
│   ├── run_ablation_batch1.sh        # 消融實驗（batch 1）
│   ├── run_ablation_batch2.sh        # 消融實驗（batch 2）
│   └── tests/                        # 各開關的 pytest 單元測試
├── Datasets/
│   ├── path_resolver.py              # ${DATASET_ROOT} / ${REPO_ROOT} 展開
│   ├── *.csv                         # 資料集 manifest（可攜佔位符）
│   └── class_presence.json           # RCS 用的逐影像類別表
└── README.md
```

未追蹤項目:模型權重（`checkpoints/`）、訓練輸出（`outputs_*/`）、特徵快取、benchmark 提交包、論文原稿。見上方各下載連結。

---

## 引用

```bibtex
@article{pairsam,
  title   = {PairSAM: Reference-Guided Adaptation of SAM for
             Adverse-Weather Semantic Segmentation},
  author  = {[Author Names]},
  journal = {[Venue]},
  year    = {2026},
  note    = {Paper link TBD}
}
```

---

## 授權

本 repo 承襲 Segment Anything 程式碼的 **Apache License 2.0**,見 [segment-anything/LICENSE](segment-anything/LICENSE)。第三方元件（CLIP、VGG、UAWarpC）保留各自授權。

## 致謝

PairSAM 建構於 **Segment Anything** 與 **Mask2Former**（Meta AI Research）、**CLIP**（OpenAI）,以及 **CMA / Refign** 的 GNSS 配對評估協定之上。感謝 **ACDC**、**Dark Zurich**、**RobotCar Correspondence** 作者釋出這些 benchmark。

經費與機構致謝:*[發表時補上]*
