#!/usr/bin/env bash
# =============================================================================
# WeatherSAM M2F「Cityscapes 端到端預訓練」(Strategy B)：
#   encoder + m2f decoder 一起訓，產出「兩者皆 co-adapt」的預權重。
#
# 與凍結版 run_m2f_cs_pretrain.sh 的唯一差異：解凍 ViT-H 讓 encoder 與
# m2f decoder 在 Cityscapes 上端到端 co-adapt（ViTDet / Mask2Former 標準做法）。
#
#   --unfreeze_encoder_blocks 32：解凍全部 32 個 ViT-H block（patch_embed/neck
#       仍凍結）。從既有 CS 域內化 encoder 出發，只補「與 m2f 頭的 co-adaptation」，
#       非從零，故不需很多 epoch。
#   --encoder_lr_scale 0.1：encoder LR = lr × 0.1 = 5e-6（decoder 的 1/10）。
#       gentle co-adapt，防止災難性遺忘 SAM/CS 特徵。★ 此為主要可調旋鈕：
#       想更保守 → 0.02（≈pretrain_cityscapes 的 1/10）；想更用力 → 0.2。
#
# 目的：得到 encoder+decoder 皆勝任 Cityscapes 任務的完整模型，使 ACDC 階段
#   可「凍結 encoder+decoder，只訓 adapter+condition」（純參數高效天氣適應）。
#
# 產物：outputs_m2f_cs_e2e/weather_sam_best_latest.pth
#   ACDC 微調：--checkpoint outputs_m2f_cs_e2e/weather_sam_best_latest.pth
#     （build 依名稱+形狀匹配，encoder 與 decoder 權重一併載入）
#
# 成本提醒：解凍 encoder → 每步需 backprop 穿 ViT-H，約慢 3-4×。從既有 CS
#   encoder 起步建議 --epochs 15 即可（非凍結版的 30）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CITYSCAPES_ENC="/home/rvl1421/SAM_research-1/segment-anything/checkpoints/cityscapes_pretrain/sam_vit_h_cityscapes_encoder_best.pth"
OUTPUT_DIR="outputs_m2f_cs_e2e"

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
  --unfreeze_encoder_blocks 32 \
  --encoder_lr_scale 0.1 \
  --weight_decay 0.05 \
  --dice_weight 5.0 \
  --cls_weight 2.0 \
  --bce_weight 5.0 \
  --no_object_weight 0.1 \
  --num_points 12544 \
  --lr 5e-5 \
  --max_norm 1.0 \
  --epochs 15 \
  --warmup_epochs 5 \
  --warmup_gate_epochs 0 \
  --decoder_lr_scale 1.0 \
  --lovasz_weight 0.0 \
  "$@"
