#!/usr/bin/env bash
# =============================================================================
# PairSAM M2F「Cityscapes 解碼端預訓練」：source-domain pre-training。
#
# 目的（2026-07-05 定案）：
#   m2f decoder stack（SimpleFPN + MSDeformAttn pixel decoder + M2FDecoder）
#   從零訓練只餵 ACDC 1600 張會嚴重過擬合（val cls 0.12→0.35）。本階段先在
#   Cityscapes train（2975 張、同 19 類）教會解碼端「任務」，ACDC 階段再學
#   「天氣補償」（adapter / CMA-ref / condition），各模組職責與資料對應。
#
# 與 run_m2f_cityscapes.sh 的差異（其餘超參完全一致，勿另調）：
#   --no-use_vgg_adapter：關閉 DeformAdapter + CMA reference 分支
#       （Cityscapes 無 clear-reference 配對；CSV 的 ref 欄為 self-reference
#         佔位，forward 不會使用）
#   --no-ref --no-cond：CS 無天氣條件；condition token 退化為可學習常數
#   --train_csv/--val_csv：Cityscapes 官方 split（2975/500）
#   --epochs 30：2975×30 與 ACDC 1600×50 樣本量級相當
#
# 產物：outputs_m2f_cs_pretrain/weather_sam_best_latest.pth
#   (1) ACDC 微調初始權重：run_m2f_cityscapes.sh 傳
#       --checkpoint outputs_m2f_cs_pretrain/weather_sam_best_latest.pth
#       （build 依名稱+形狀匹配載入全模型，經 Trainer-ckpt 路徑，零改動）
#   (2) 論文 "Source model" baseline：本模型直接在 ACDC val/test 評測
#       即 Cityscapes-only 未適配 baseline（對齊 ACDC TPAMI Table 3 用法）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CITYSCAPES_ENC="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/cityscapes_pretrain/sam_vit_h_cityscapes_encoder_best.pth"
OUTPUT_DIR="outputs_m2f_cs_pretrain"

python -u train.py \
  --model_type vit_h \
  --decoder m2f \
  --checkpoint "$CITYSCAPES_ENC" \
  --output_dir "$OUTPUT_DIR" \
  --train_csv /home/rvl1421/SAM_research-1/Datasets/cityscapes_m2f_train.csv \
  --val_csv /home/rvl1421/SAM_research-1/Datasets/cityscapes_m2f_val.csv \
  --no-use_vgg_adapter \
  --no-ref \
  --no-cond \
  --weight_decay 0.05 \
  --dice_weight 5.0 \
  --cls_weight 2.0 \
  --bce_weight 5.0 \
  --no_object_weight 0.1 \
  --num_points 12544 \
  --lr 5e-5 \
  --max_norm 1.0 \
  --epochs 30 \
  --warmup_epochs 5 \
  --warmup_gate_epochs 0 \
  --decoder_lr_scale 1.0 \
  --lovasz_weight 0.0 \
  "$@"
